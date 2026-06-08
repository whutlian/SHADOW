from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LAD_STAGE_DEFAULTS: dict[str, Any] = {
    "stage": "lad",
    "seed": 42,
    "diffusion_enabled": False,
    "diffusion_status": "diagnostic_only",
    "feature_mode": "label_affinity",
    "label_affinity": True,
    "label_affinity_mode": "all",
    "label_affinity_self_exclude": True,
    "label_affinity_block_norm": "row_l1",
    "compiled_head": True,
    "compiled_head_fusion": "concat_mlp",
    "compiled_hidden_dim": 256,
    "compiled_dropout": 0.3,
    "compiled_block_gate": True,
    "boundary_prototypes": False,
}


@dataclass(frozen=True)
class LADVariant:
    name: str
    compiled_head: bool
    label_affinity: bool
    boundary_prototypes: bool


LAD_VARIANTS: tuple[LADVariant, ...] = (
    LADVariant("V0_current_best", compiled_head=False, label_affinity=False, boundary_prototypes=False),
    LADVariant("V1_compiled_demand_head", compiled_head=True, label_affinity=False, boundary_prototypes=False),
    LADVariant("V2_compiled_plus_lad", compiled_head=True, label_affinity=True, boundary_prototypes=False),
    LADVariant("V3_compiled_lad_boundary", compiled_head=True, label_affinity=True, boundary_prototypes=True),
)


SMALL_RATIOS: dict[str, list[float]] = {
    "acm": [0.096],
    "dblp": [0.005, 0.065],
    "imdb": [0.005, 0.025, 0.05],
}


MEDIUM_RATIOS: dict[str, list[float]] = {
    "ogbn-arxiv": [0.06, 0.12],
    "ogbn-products": [0.06, 0.12],
}


DATASET_LOSS: dict[str, str] = {
    "acm": "clipped",
    "dblp": "clipped",
    "imdb": "class_balanced",
    "ogbn-arxiv": "sqrt_weighted_logit_adjusted",
    "ogbn-products": "sqrt_weighted_logit_adjusted",
}


def lad_feature_mode(*, label_affinity: bool, metapath: bool = False) -> str:
    if label_affinity and metapath:
        return "label_affinity_metapath"
    if label_affinity:
        return "label_affinity"
    if metapath:
        return "metapath"
    return "base"


def ratio_label(ratio: float) -> str:
    return str(float(ratio)).replace(".", "p")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]], required_fields: Iterable[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(required_fields or [])
    seen = set(fields)
    for key in sorted({key for row in rows for key in row}):
        if key not in seen:
            fields.append(key)
            seen.add(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


LAD_TABLE_FIELDS = [
    "dataset",
    "ratio",
    "variant",
    "seed",
    "status",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "condensed_nodes_total",
    "condensed_edges_total",
    "total_condensed_node_ratio",
    "byte_size_compression",
    "compiled_head",
    "label_affinity",
    "boundary_prototypes",
    "loss_type",
    "train_time_s",
    "inference_time_s",
    "lad_precompute_time_s",
    "peak_cpu_ram_mb",
    "peak_gpu_ram_mb",
]


def summary_to_lad_row(summary: dict[str, Any], *, dataset: str, ratio: float, variant: str, log_path: Path) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "ratio": ratio,
        "variant": variant,
        "seed": summary.get("seed", 42),
        "status": summary.get("status", "completed"),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "byte_size_compression": summary.get("byte_size_compression", ""),
        "compiled_head": summary.get("compiled_head", False),
        "label_affinity": summary.get("label_affinity", False),
        "boundary_prototypes": summary.get("boundary_prototypes", False),
        "loss_type": summary.get("loss_type", ""),
        "train_time_s": summary.get("training_time", summary.get("train_time_s", "")),
        "inference_time_s": summary.get("inference_time", summary.get("inference_time_s", "")),
        "lad_precompute_time_s": summary.get("lad_precompute_time_s", ""),
        "peak_cpu_ram_mb": summary.get("peak_cpu_ram_mb", summary.get("peak_cpu_ram", "")),
        "peak_gpu_ram_mb": summary.get("peak_gpu_ram_mb", summary.get("peak_gpu_ram", "")),
        "source_log": str(log_path),
        "reason": summary.get("reason", ""),
    }
