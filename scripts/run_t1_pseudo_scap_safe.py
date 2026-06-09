from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.run_t1_logit_correct_safe import _all_logits, _labels_all
from scripts.t1_safe_common import read_csv, target_edge_for_cache
from shadow_hgc.features.pseudo_scap import target_target_pseudo_scap
from shadow_hgc.logits import load_logits_cache
from shadow_hgc.logits.correct_smooth import select_best_validation_row
from shadow_hgc.logits.pseudo_scap import build_t1_pseudo_labels
from shadow_hgc.logits.replay import metrics_from_logits


FIELDS = [
    "dataset",
    "base_variant",
    "threshold",
    "pseudo_weight",
    "temperature",
    "affinity_lambda",
    "prior_centering",
    "pseudo_coverage",
    "pseudo_mean_confidence",
    "train_label_class_distribution",
    "pseudo_class_distribution",
    "valid_acc_before",
    "valid_acc_after",
    "test_acc_before",
    "test_acc_after",
    "macro_f1_before",
    "macro_f1_after",
    "predicted_class_count_before",
    "predicted_class_count_after",
    "promotion_status",
    "promotion_reason",
    "uses_diffusion",
    "uses_dense_p2",
    "uses_bounded_edges",
]


def _final_logits(cache, logits: torch.Tensor, edge_index: torch.Tensor, spec: dict) -> tuple[torch.Tensor, dict]:
    labels = torch.from_numpy(_labels_all(cache)).to(torch.long)
    pseudo = build_t1_pseudo_labels(
        logits,
        labels=labels,
        train_idx=torch.from_numpy(np.asarray(cache.train_idx).copy()).to(torch.long),
        threshold=spec["threshold"],
        pseudo_weight=spec["pseudo_weight"],
        temperature=spec["temperature"],
    )
    affinity, diagnostics = target_target_pseudo_scap(edge_index=edge_index, pseudo=pseudo.pseudo, weights=pseudo.weights, num_nodes=logits.shape[0])
    probs = torch.softmax(logits, dim=1)
    mixed = (1.0 - float(spec["affinity_lambda"])) * probs + float(spec["affinity_lambda"]) * affinity
    mixed = mixed.clamp_min(1e-12)
    mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.log(mixed), {**pseudo.diagnostics, **diagnostics}


def _blocked(row: dict, reason: str) -> dict:
    return {
        "dataset": row.get("dataset", ""),
        "base_variant": row.get("base_variant", ""),
        "promotion_status": "blocked",
        "promotion_reason": reason,
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
    }


def _run_row(row: dict, args) -> list[dict]:
    replay = load_logits_cache(row["cache_path"])
    gate = load_logits_cache(row["gate_cache_path"]) if row.get("gate_cache_path") else None
    if gate is None or gate.valid_idx is None:
        return [_blocked(row, "validation_protocol_unavailable")]
    edge_index = target_edge_for_cache(row["cache_path"])
    if edge_index is None:
        return [_blocked(row, "no_target_target_relation")]
    gate_logits = _all_logits(gate)
    replay_logits = _all_logits(replay)
    labels_gate = _labels_all(gate)
    valid_idx = torch.from_numpy(np.asarray(gate.valid_idx).copy()).to(torch.long)
    test_idx = torch.from_numpy(np.asarray(replay.test_idx).copy()).to(torch.long)
    base_valid = metrics_from_logits(gate_logits[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
    base_test = metrics_from_logits(replay.test_logits, replay.y_test, num_classes=replay.meta.num_classes)
    candidates = []
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        for pseudo_weight in [0.25, 0.5, 1.0]:
            for temperature in [1.0, 2.0]:
                for affinity_lambda in [0.05, 0.1, 0.2, 0.4, 0.6]:
                    for prior_centering in [False]:
                        spec = {
                            "threshold": threshold,
                            "pseudo_weight": pseudo_weight,
                            "temperature": temperature,
                            "affinity_lambda": affinity_lambda,
                            "prior_centering": prior_centering,
                        }
                        out, diagnostics = _final_logits(gate, gate_logits, edge_index, spec)
                        valid_metrics = metrics_from_logits(out[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
                        candidates.append({**spec, **diagnostics, "valid_acc": valid_metrics["accuracy"], "valid_macro_f1": valid_metrics["macro_f1"]})
    selected = select_best_validation_row(candidates)
    replay_out, diagnostics = _final_logits(replay, replay_logits, edge_index, selected)
    test_metrics = metrics_from_logits(replay_out[test_idx], replay.y_test, num_classes=replay.meta.num_classes)
    valid_improved = float(selected["valid_acc"]) > float(base_valid["accuracy"])
    class_ok = int(test_metrics["predicted_class_count"]) >= int(base_test["predicted_class_count"])
    macro_ok = float(test_metrics["macro_f1"]) >= float(base_test["macro_f1"]) - float(args.macro_f1_tolerance)
    reasons = []
    if not valid_improved:
        reasons.append("validation_no_improvement")
    if not class_ok:
        reasons.append("predicted_class_count_collapse")
    if not macro_ok:
        reasons.append("macro_f1_regression")
    status = "promoted" if not reasons else "blocked"
    train_hist = torch.bincount(torch.from_numpy(_labels_all(replay))[torch.from_numpy(np.asarray(replay.train_idx).copy())].clamp_min(0), minlength=replay.meta.num_classes).tolist()
    pseudo_hist = torch.bincount(build_t1_pseudo_labels(replay_logits, labels=torch.from_numpy(_labels_all(replay)), train_idx=torch.from_numpy(np.asarray(replay.train_idx).copy()), threshold=selected["threshold"], pseudo_weight=selected["pseudo_weight"], temperature=selected["temperature"]).pseudo.argmax(dim=1), minlength=replay.meta.num_classes).tolist()
    return [
        {
            "dataset": row["dataset"],
            "base_variant": row["base_variant"],
            "threshold": selected["threshold"],
            "pseudo_weight": selected["pseudo_weight"],
            "temperature": selected["temperature"],
            "affinity_lambda": selected["affinity_lambda"],
            "prior_centering": selected["prior_centering"],
            "pseudo_coverage": selected["pseudo_coverage"],
            "pseudo_mean_confidence": selected["mean_confidence"],
            "train_label_class_distribution": train_hist,
            "pseudo_class_distribution": pseudo_hist,
            "valid_acc_before": base_valid["accuracy"],
            "valid_acc_after": selected["valid_acc"],
            "test_acc_before": base_test["accuracy"],
            "test_acc_after": test_metrics["accuracy"],
            "macro_f1_before": base_test["macro_f1"],
            "macro_f1_after": test_metrics["macro_f1"],
            "predicted_class_count_before": base_test["predicted_class_count"],
            "predicted_class_count_after": test_metrics["predicted_class_count"],
            "promotion_status": status,
            "promotion_reason": "validation_selected" if status == "promoted" else ",".join(reasons),
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "uses_bounded_edges": False,
        }
    ]


def run(args) -> list[dict]:
    rows = []
    for row in read_csv(args.replay_audit):
        if row.get("cache_status") == "available_verified" and row.get("cache_path"):
            rows.extend(_run_row(row, args))
        else:
            rows.append(_blocked(row, row.get("blocked_reason") or row.get("cache_status", "cache_not_verified")))
    write_csv(args.output, rows, FIELDS)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# T1.1 Pseudo-SCAP Safe Summary\n\n" + f"- CSV: `{args.output}`\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe Pseudo-SCAP.")
    parser.add_argument("--replay-audit", default="experiments/tables/t1_cache_replay_audit_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_pseudo_scap_safe_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_pseudo_scap_safe_summary.md")
    parser.add_argument("--macro-f1-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
