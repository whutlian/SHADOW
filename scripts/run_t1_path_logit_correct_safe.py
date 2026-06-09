from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.run_t1_logit_correct_safe import _all_logits, _labels_all
from scripts.t1_safe_common import read_csv
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.logits import load_logits_cache
from shadow_hgc.logits.path_correct import PathStep, apply_path_logit_correct
from shadow_hgc.logits.replay import metrics_from_logits


FIELDS = [
    "dataset",
    "base_variant",
    "path_set",
    "alpha",
    "space",
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
    "uses_dense_path_adjacency",
    "exposes_metapath_edge_type",
]


def _candidate_paths(dataset: str) -> list[tuple[str, list[PathStep]]]:
    graph = load_processed_small_dataset(dataset)
    paths: list[tuple[str, list[PathStep]]] = []
    for relation in graph.relations:
        if relation.destination_type != graph.target_type or relation.source_type == graph.target_type:
            continue
        edge_index = graph.edge_index[relation].to(torch.long)
        reverse = torch.stack([edge_index[1], edge_index[0]], dim=0).contiguous()
        name = f"{graph.target_type}-{relation.source_type}-{graph.target_type}"
        paths.append(
            (
                name,
                [
                    PathStep(
                        edge_index=reverse,
                        num_src=int(graph.num_nodes[graph.target_type]),
                        num_dst=int(graph.num_nodes[relation.source_type]),
                        name=f"{graph.target_type}_to_{relation.source_type}",
                    ),
                    PathStep(
                        edge_index=edge_index,
                        num_src=int(graph.num_nodes[relation.source_type]),
                        num_dst=int(graph.num_nodes[graph.target_type]),
                        name=f"{relation.source_type}_to_{graph.target_type}",
                    ),
                ],
            )
        )
    return paths


def _select_and_apply(row: dict, args) -> dict:
    replay = load_logits_cache(row["cache_path"])
    gate = load_logits_cache(row["gate_cache_path"])
    if gate.valid_idx is None:
        return {**_blocked(row, "validation_protocol_unavailable"), "path_set": ""}
    path_specs = _candidate_paths(row["dataset"])
    if not path_specs:
        return {**_blocked(row, "no_schema_default_two_step_path"), "path_set": ""}

    gate_logits = _all_logits(gate)
    replay_logits = _all_logits(replay)
    labels_gate = _labels_all(gate)
    labels_replay = _labels_all(replay)
    valid_idx = torch.from_numpy(np.asarray(gate.valid_idx).copy()).to(torch.long)
    test_idx = torch.from_numpy(np.asarray(replay.test_idx).copy()).to(torch.long)
    base_valid = metrics_from_logits(gate_logits[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
    base_test = metrics_from_logits(replay.test_logits, replay.y_test, num_classes=replay.meta.num_classes)

    candidates: list[dict] = []
    for path_name, steps in path_specs:
        for space in ["prob", "logit"]:
            for alpha in [0.05, 0.1, 0.2, 0.4, 0.6]:
                result = apply_path_logit_correct(base_logits=gate_logits, steps=steps, alpha=alpha, space=space)
                valid_metrics = metrics_from_logits(result.logits[valid_idx], labels_gate[gate.valid_idx], num_classes=gate.meta.num_classes)
                candidates.append(
                    {
                        "path_set": path_name,
                        "steps": steps,
                        "alpha": alpha,
                        "space": space,
                        "valid_acc": valid_metrics["accuracy"],
                        "valid_macro_f1": valid_metrics["macro_f1"],
                        "diagnostics": result.diagnostics,
                    }
                )
    selected = max(candidates, key=lambda item: (float(item["valid_acc"]), float(item["valid_macro_f1"])))
    replay_out = apply_path_logit_correct(
        base_logits=replay_logits,
        steps=selected["steps"],
        alpha=float(selected["alpha"]),
        space=str(selected["space"]),
    )
    test_metrics = metrics_from_logits(replay_out.logits[test_idx], labels_replay[replay.test_idx], num_classes=replay.meta.num_classes)
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
    return {
        "dataset": row["dataset"],
        "base_variant": row["base_variant"],
        "path_set": selected["path_set"],
        "alpha": selected["alpha"],
        "space": selected["space"],
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
        "uses_dense_path_adjacency": False,
        "exposes_metapath_edge_type": False,
    }


def run(args) -> list[dict]:
    rows = []
    for row in read_csv(args.replay_audit):
        if row["dataset"] in {"dblp", "imdb"} and row.get("cache_status") == "available_verified":
            rows.append(_select_and_apply(row, args))
        elif row["dataset"] in {"dblp", "imdb"}:
            rows.append({**_blocked(row, row.get("blocked_reason") or row.get("cache_status", "cache_not_verified")), "path_set": ""})
    if not rows:
        rows.append(_blocked({"dataset": "dblp", "base_variant": "R+ relation-linear current-best"}, "cache_not_verified"))
        rows.append(_blocked({"dataset": "imdb", "base_variant": "clean S1 MAM/MDM/MKM"}, "cache_not_verified"))
    write_csv(args.output, rows, FIELDS)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    promoted = [row for row in rows if row.get("promotion_status") == "promoted"]
    Path(args.report).write_text(
        "# T1.1 PathLogitCorrectLite Summary\n\n"
        f"- Promoted rows: `{len(promoted)}`\n"
        f"- CSV: `{args.output}`\n",
        encoding="utf-8",
    )
    return rows


def _blocked(row: dict, reason: str) -> dict:
    return {
        "dataset": row.get("dataset", ""),
        "base_variant": row.get("base_variant", ""),
        "alpha": "",
        "space": "",
        "valid_acc_before": "",
        "valid_acc_after": "",
        "valid_macro_f1_after": "",
        "test_acc_before": "",
        "test_acc_after": "",
        "macro_f1_before": "",
        "macro_f1_after": "",
        "predicted_class_count_before": "",
        "predicted_class_count_after": "",
        "promotion_status": "blocked",
        "promotion_reason": reason,
        "uses_dense_path_adjacency": False,
        "exposes_metapath_edge_type": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe PathLogitCorrectLite.")
    parser.add_argument("--replay-audit", default="experiments/tables/t1_cache_replay_audit_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_path_logit_correct_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_path_logit_correct_summary.md")
    parser.add_argument("--macro-f1-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
