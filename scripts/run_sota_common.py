from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.run_lad_common import DATASET_LOSS, ratio_label


SMALL_SOTA_RATIOS: dict[str, list[float]] = {
    "acm": [0.012, 0.024, 0.048, 0.096],
    "dblp": [0.012, 0.024, 0.048, 0.096],
    "imdb": [0.012, 0.024, 0.048, 0.096],
}

MEDIUM_SOTA_FULL_NODE_RATIOS: dict[str, list[float]] = {
    "ogbn-arxiv": [0.0005, 0.0025, 0.005],
    "ogbn-products": [0.0005, 0.0025, 0.005],
}

PATH_LAD_BLOCKS: dict[str, list[str]] = {
    "acm": ["PAP", "PSP", "PTP"],
    "dblp": ["APA", "APVPA", "APTPA"],
    "imdb": ["MAM", "MDM", "MKM"],
    "ogbn-arxiv": ["P1"],
    "ogbn-products": ["P1"],
}


@dataclass(frozen=True)
class SOTAVariant:
    name: str
    compiled_head: bool
    label_affinity: bool
    path_label_affinity: bool
    prototype_mode: str
    source_anchor_mode: str
    teacher_type: str
    use_kd: bool
    boundary_prototypes: bool = False


SOTA_SMALL_VARIANTS: tuple[SOTAVariant, ...] = (
    SOTAVariant("S0_current_best", False, False, False, "kmeans_mean", "none", "none", False),
    SOTAVariant("S1_sehgnn_lite_metapath", True, False, False, "kmeans_mean", "none", "none", False),
    SOTAVariant("S2_coverage_medoids", True, False, False, "coverage_medoid", "none", "none", False),
    SOTAVariant("S3_path_lad_source_anchor", True, False, True, "coverage_medoid", "coverage_residual", "none", False),
    SOTAVariant("S4_teacher_kd", True, False, True, "coverage_medoid", "coverage_residual", "sehgnn_lite", True),
)

SOTA_MEDIUM_VARIANTS: tuple[SOTAVariant, ...] = (
    SOTAVariant("S0_current_best", False, False, False, "kmeans_mean", "none", "none", False),
    SOTAVariant("S2_coverage_medoids", True, True, False, "coverage_medoid", "none", "none", False),
    SOTAVariant("S4_teacher_kd", True, True, True, "coverage_medoid", "none", "sign_lad_mlp", True),
)

SOTA_TABLE_FIELDS = [
    "dataset",
    "variant",
    "seed",
    "requested_ratio",
    "requested_full_condensed_node_ratio",
    "effective_target_ratio",
    "total_condensed_node_ratio",
    "total_condensed_edge_ratio",
    "byte_size_compression",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "prototype_mode",
    "model_type",
    "metapath_blocks",
    "path_lad_blocks",
    "source_anchor_mode",
    "teacher_type",
    "use_kd",
    "condense_time_s",
    "train_time_s",
    "infer_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "compiled_block_stats_source",
    "status",
    "reason",
    "source_log",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: Iterable[str] = SOTA_TABLE_FIELDS) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    seen = set(field_list)
    for key in sorted({key for row in rows for key in row}):
        if key not in seen:
            field_list.append(key)
            seen.add(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_to_sota_row(
    summary: dict[str, Any],
    *,
    dataset: str,
    variant: SOTAVariant | str,
    log_path: Path,
    requested_ratio: float | None = None,
    requested_full_ratio: float | None = None,
) -> dict[str, Any]:
    variant_name = variant if isinstance(variant, str) else variant.name
    return {
        "dataset": dataset,
        "variant": variant_name,
        "seed": summary.get("seed", 42),
        "requested_ratio": "" if requested_ratio is None else requested_ratio,
        "requested_full_condensed_node_ratio": "" if requested_full_ratio is None else requested_full_ratio,
        "effective_target_ratio": summary.get("effective_target_ratio", ""),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "total_condensed_edge_ratio": summary.get("total_condensed_edge_ratio", ""),
        "byte_size_compression": summary.get("byte_size_compression", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "prototype_mode": summary.get("prototype_mode", ""),
        "model_type": summary.get("model_type", summary.get("model", "")),
        "metapath_blocks": json.dumps(
            summary.get("metapath_blocks")
            or summary.get("multiscale_metadata", {}).get("metapath_names")
            or summary.get("multiscale_metadata", {}).get("blocks", [])
        ),
        "path_lad_blocks": json.dumps(summary.get("path_lad_blocks", [])),
        "source_anchor_mode": summary.get("source_anchor_mode", ""),
        "teacher_type": summary.get("teacher_type", summary.get("teacher", {}).get("type", "")),
        "use_kd": summary.get("use_kd", summary.get("teacher", {}).get("use_kd", False)),
        "condense_time_s": summary.get("condensation_time", ""),
        "train_time_s": summary.get("training_time", summary.get("train_time_s", "")),
        "infer_time_s": summary.get("inference_time", summary.get("infer_time_s", "")),
        "peak_cpu_ram_gb": (
            float(summary["peak_cpu_ram"]) / (1024**3)
            if summary.get("peak_cpu_ram", "") != ""
            else ""
        ),
        "peak_gpu_ram_gb": (
            float(summary["peak_gpu_ram"]) / (1024**3)
            if summary.get("peak_gpu_ram", "") != ""
            else ""
        ),
        "compiled_block_stats_source": summary.get("compiled_block_stats_source", ""),
        "status": summary.get("status", "completed"),
        "reason": summary.get("reason", ""),
        "source_log": str(log_path),
    }


def sota_loss(dataset: str, variant: SOTAVariant) -> str:
    if variant.use_kd and dataset in {"ogbn-arxiv", "ogbn-products"}:
        return "sqrt_weighted_logit_adjusted"
    return DATASET_LOSS[dataset]


def sota_ratio_label(ratio: float) -> str:
    return ratio_label(ratio)
