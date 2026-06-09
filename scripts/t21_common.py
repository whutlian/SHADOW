from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.t2_common import SAFE_BASELINES


T21_FORBIDDEN_FLAGS = [
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "uses_diffusion_legacy",
]


T21_PREPROP_FIELDS = [
    "dataset",
    "target_type",
    "status",
    "manifest_dir",
    "num_blocks",
    "block_names",
    "total_cache_bytes",
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "feature_dim",
    "uses_memmap",
    *T21_FORBIDDEN_FLAGS,
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "reason",
]


T21_FULLGRAPH_FIELDS = [
    "dataset",
    "row_kind",
    "model_type",
    "selection_protocol",
    "selection_score",
    "selected_blocks",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "valid_acc",
    "valid_macro_f1",
    "class_coverage",
    "safe_baseline",
    "safe_baseline_acc",
    "safe_baseline_macro_f1",
    "delta_acc_vs_safe",
    "delta_macro_f1_vs_safe",
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "feature_dim",
    "num_blocks",
    "cache_bytes",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "uses_memmap",
    *T21_FORBIDDEN_FLAGS,
    "uses_full_graph_backprop",
    "uses_train_labels_only",
    "source_log",
]


T21_PRODUCTS_FIELDS = [
    "dataset",
    "target_type",
    "status",
    "reason",
    "run_mode",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "selected_blocks",
    "preprop_blocks",
    "manifest_dir",
    "total_cache_bytes",
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "feature_dim",
    "training_epochs",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    *T21_FORBIDDEN_FLAGS,
]


T21_RECOVERY_FIELDS = [
    "dataset",
    "recovery_row",
    "fullgraph_status",
    "fullgraph_accuracy",
    "fullgraph_macro_f1",
    "selected_blocks",
    "status",
    "promoted",
    "full_to_identity_gap",
    "identity_to_oracle_gap",
    "oracle_to_shadow_gap",
    "full_to_shadow_gap",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "reason",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    target.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    return target


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def no_forbidden_flags(row: dict[str, Any]) -> bool:
    return not any(bool_value(row.get(flag, False)) for flag in T21_FORBIDDEN_FLAGS)


def t21_safe_baseline(dataset: str) -> dict[str, Any]:
    return SAFE_BASELINES[dataset]
