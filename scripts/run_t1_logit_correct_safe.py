from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.t1_safe_common import read_csv, target_edge_for_cache
from shadow_hgc.logits import load_logits_cache
from shadow_hgc.logits.correct_smooth import correct_and_smooth_probabilities, select_best_validation_row, smooth_probabilities
from shadow_hgc.logits.replay import metrics_from_logits


FIELDS = [
    "dataset",
    "base_variant",
    "mode",
    "correct_alpha",
    "correct_steps",
    "smooth_alpha",
    "smooth_steps",
    "valid_acc_before",
    "valid_acc_after",
    "valid_macro_f1_after",
    "test_acc_before",
    "test_acc_after",
    "macro_f1_before",
    "macro_f1_after",
    "predicted_class_count_before",
    "predicted_class_count_after",
    "promotion_status",
    "promotion_reason",
    "validation_selected",
    "uses_diffusion",
    "uses_dense_p2",
    "uses_bounded_edges",
    "cache_path",
]


def _all_logits(cache) -> torch.Tensor:
    if cache.all_target_logits is None:
        raise ValueError("cache missing all_target_logits")
    return torch.from_numpy(np.asarray(cache.all_target_logits).copy()).to(torch.float32)


def _run_spec(cache, logits: torch.Tensor, edge_index: torch.Tensor, spec: dict) -> torch.Tensor:
    if spec["mode"] == "smooth_prob":
        probs = torch.softmax(logits, dim=1)
        return torch.log(
            smooth_probabilities(
                probabilities=probs,
                edge_index=edge_index,
                num_nodes=logits.shape[0],
                alpha=spec["smooth_alpha"],
                steps=spec["smooth_steps"],
            ).clamp_min(1e-12)
        )
    return correct_and_smooth_probabilities(
        logits=logits,
        labels=torch.from_numpy(cache.y_train if cache.valid_idx is None else np.asarray(cache.y_train).copy()).new_tensor(
            np.asarray(_labels_all(cache)).copy(), dtype=torch.long
        ),
        train_idx=torch.from_numpy(np.asarray(cache.train_idx).copy()).to(torch.long),
        edge_index=edge_index,
        num_nodes=logits.shape[0],
        correct_alpha=spec["correct_alpha"],
        correct_steps=spec["correct_steps"],
        smooth_alpha=spec["smooth_alpha"],
        smooth_steps=spec["smooth_steps"],
    ).logits


def _labels_all(cache) -> np.ndarray:
    labels = np.full((int(cache.meta.num_target_nodes),), -1, dtype=np.int64)
    labels[np.asarray(cache.train_idx, dtype=np.int64)] = np.asarray(cache.y_train, dtype=np.int64)
    if cache.valid_idx is not None and cache.y_valid is not None:
        labels[np.asarray(cache.valid_idx, dtype=np.int64)] = np.asarray(cache.y_valid, dtype=np.int64)
    if cache.test_idx is not None and cache.y_test is not None:
        labels[np.asarray(cache.test_idx, dtype=np.int64)] = np.asarray(cache.y_test, dtype=np.int64)
    return labels


def _select_and_apply(row: dict, args) -> list[dict]:
    replay = load_logits_cache(row["cache_path"])
    gate = load_logits_cache(row["gate_cache_path"]) if row.get("gate_cache_path") else None
    if gate is None or gate.valid_idx is None:
        return [{**_blocked(row, "validation_protocol_unavailable"), "cache_path": row["cache_path"]}]
    edge_index = target_edge_for_cache(row["cache_path"])
    if edge_index is None:
        return [{**_blocked(row, "no_target_target_relation"), "cache_path": row["cache_path"]}]
    gate_logits = _all_logits(gate)
    replay_logits = _all_logits(replay)
    labels_gate = _labels_all(gate)
    labels_replay = _labels_all(replay)
    valid_idx = torch.from_numpy(np.asarray(gate.valid_idx).copy()).to(torch.long)
    test_idx = torch.from_numpy(np.asarray(replay.test_idx).copy()).to(torch.long)
    base_valid = metrics_from_logits(gate_logits[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
    base_test = metrics_from_logits(replay.test_logits, replay.y_test, num_classes=replay.meta.num_classes)
    candidate_rows: list[dict] = []
    candidate_specs: list[dict] = []
    for smooth_alpha in [0.05, 0.1, 0.2, 0.4]:
        for smooth_steps in [1, 2, 4]:
            candidate_specs.append({"mode": "smooth_prob", "correct_alpha": "", "correct_steps": "", "smooth_alpha": smooth_alpha, "smooth_steps": smooth_steps})
    for correct_alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for correct_steps in [1, 2, 4, 8]:
            for smooth_alpha in [0.0, 0.05, 0.1, 0.2, 0.4]:
                for smooth_steps in [0, 1, 2, 4]:
                    candidate_specs.append({"mode": "correct_smooth", "correct_alpha": correct_alpha, "correct_steps": correct_steps, "smooth_alpha": smooth_alpha, "smooth_steps": smooth_steps})
    for spec in candidate_specs:
        try:
            gate_out = _run_spec(gate, gate_logits, edge_index, spec)
        except Exception as exc:
            continue
        valid_metrics = metrics_from_logits(gate_out[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
        candidate_rows.append({**spec, "valid_acc": valid_metrics["accuracy"], "valid_macro_f1": valid_metrics["macro_f1"]})
    if not candidate_rows:
        return [{**_blocked(row, "no_valid_candidate"), "cache_path": row["cache_path"]}]
    selected = select_best_validation_row(candidate_rows)
    replay_out = _run_spec(replay, replay_logits, edge_index, selected)
    test_metrics = metrics_from_logits(replay_out[test_idx], replay.y_test, num_classes=replay.meta.num_classes)
    valid_improved = float(selected["valid_acc"]) > float(base_valid["accuracy"])
    class_ok = int(test_metrics["predicted_class_count"]) >= int(base_test["predicted_class_count"])
    macro_ok = float(test_metrics["macro_f1"]) >= float(base_test["macro_f1"]) - float(args.macro_f1_tolerance)
    status = "promoted" if valid_improved and class_ok and macro_ok else "blocked"
    reasons = []
    if not valid_improved:
        reasons.append("validation_no_improvement")
    if not class_ok:
        reasons.append("predicted_class_count_collapse")
    if not macro_ok:
        reasons.append("macro_f1_regression")
    return [
        {
            "dataset": row["dataset"],
            "base_variant": row["base_variant"],
            "mode": selected["mode"],
            "correct_alpha": selected.get("correct_alpha", ""),
            "correct_steps": selected.get("correct_steps", ""),
            "smooth_alpha": selected.get("smooth_alpha", ""),
            "smooth_steps": selected.get("smooth_steps", ""),
            "valid_acc_before": base_valid["accuracy"],
            "valid_acc_after": selected["valid_acc"],
            "valid_macro_f1_after": selected["valid_macro_f1"],
            "test_acc_before": base_test["accuracy"],
            "test_acc_after": test_metrics["accuracy"],
            "macro_f1_before": base_test["macro_f1"],
            "macro_f1_after": test_metrics["macro_f1"],
            "predicted_class_count_before": base_test["predicted_class_count"],
            "predicted_class_count_after": test_metrics["predicted_class_count"],
            "promotion_status": status,
            "promotion_reason": "validation_selected" if status == "promoted" else ",".join(reasons),
            "validation_selected": True,
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "uses_bounded_edges": False,
            "cache_path": row["cache_path"],
        }
    ]


def _blocked(row: dict, reason: str) -> dict:
    return {
        "dataset": row.get("dataset", ""),
        "base_variant": row.get("base_variant", ""),
        "mode": "blocked",
        "promotion_status": "blocked",
        "promotion_reason": reason,
        "validation_selected": False,
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
    }


def run(args) -> list[dict]:
    rows = []
    for row in read_csv(args.replay_audit):
        if row.get("cache_status") == "available_verified" and row.get("cache_path"):
            rows.extend(_select_and_apply(row, args))
        else:
            rows.append(_blocked(row, row.get("blocked_reason") or row.get("cache_status", "cache_not_verified")))
    write_csv(args.output, rows, FIELDS)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# T1.1 Safe LogitCorrectLite Summary\n\n" + f"- CSV: `{args.output}`\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe LogitCorrectLite.")
    parser.add_argument("--replay-audit", default="experiments/tables/t1_cache_replay_audit_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_safe_logit_correct_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_safe_logit_correct_summary.md")
    parser.add_argument("--macro-f1-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
