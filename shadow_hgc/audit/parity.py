from __future__ import annotations

from typing import Any

from shadow_hgc.audit.schema_checks import coerce_list


FULLGRAPH_PARITY_REQUIRED_FIELDS = [
    "dataset",
    "variant",
    "seed",
    "target_type",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "target_gate",
    "gate_passed",
    "blocked_by_fullgraph_backbone",
    "model_type",
    "feature_mode",
    "metapath_blocks",
    "path_lad_blocks",
    "num_nodes_by_type",
    "num_edges_by_type",
    "split_hash",
    "feature_hash",
    "label_hash",
    "schema_hash",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "num_classes",
    "train_class_count",
    "valid_class_count",
    "test_class_count",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_mb",
    "peak_gpu_ram_mb",
]


def _missing(row: dict, field: str) -> bool:
    return field not in row or row.get(field) in ("", None)


def validate_fullgraph_parity_row(row: dict[str, Any]) -> dict:
    reasons = [f"{field}_missing" for field in FULLGRAPH_PARITY_REQUIRED_FIELDS if _missing(row, field)]
    if str(row.get("variant", "")).startswith("fullgraph") and row.get("status") == "completed":
        for field in ("split_hash", "feature_hash", "label_hash", "schema_hash"):
            if _missing(row, field) and f"{field}_missing" not in reasons:
                reasons.append(f"{field}_missing")
    return {"valid": not reasons, "reasons": reasons, "warnings": []}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}
    return bool(value)


def validate_promoted_row(row: dict[str, Any]) -> dict:
    variant = str(row.get("variant", "")).lower()
    dataset = str(row.get("dataset", "")).lower()
    reasons: list[str] = []
    warnings: list[str] = []
    if _truthy(row.get("use_diffusion", row.get("diffusion_enabled", False))):
        reasons.append("diffusion_not_allowed_in_promoted_path")
    path_lad_blocks = [str(value).upper() for value in coerce_list(row.get("path_lad_blocks"))]
    if dataset == "ogbn-products" and ("P2" in path_lad_blocks or "two_hop" in variant):
        reasons.append("products_p2_lad_not_allowed_in_promoted_path")
    if "sehgnn" in variant and str(row.get("model_type", "")).lower() != "sehgnn_lite":
        reasons.append("sehgnn_row_requires_model_type_sehgnn_lite")
    if "metapath" in variant and not coerce_list(row.get("metapath_blocks")):
        reasons.append("metapath_blocks_missing")
    if "kd" in variant and _missing(row, "teacher_val_acc"):
        reasons.append("kd_teacher_val_acc_missing")
    if "pathlad" in variant or "path_lad" in variant:
        for field in ("path_lad_row_normalize", "path_lad_leave_one_out", "path_lad_hub_clip_quantile"):
            if _missing(row, field):
                reasons.append(f"{field}_missing")
    if str(row.get("loader_mode", "")) == "full_schema" and row.get("schema_required_edges_present") is False:
        reasons.append("full_schema_required_edges_absent")
    return {"valid": not reasons, "reasons": reasons, "warnings": warnings}

