from __future__ import annotations

import math
from typing import Any

from shadow_hgc.audit.schema_checks import coerce_dict, coerce_list, require_nonempty_feature_blocks


def _variant_name(config: dict) -> str:
    return str(config.get("variant", config.get("name", ""))).lower()


def _field(config: dict, runtime: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in config and config[name] not in ("", None):
            return config[name]
        if name in runtime and runtime[name] not in ("", None):
            return runtime[name]
    teacher = config.get("teacher")
    if isinstance(teacher, dict):
        for name in names:
            if name in teacher and teacher[name] not in ("", None):
                return teacher[name]
    teacher = runtime.get("teacher")
    if isinstance(teacher, dict):
        for name in names:
            if name in teacher and teacher[name] not in ("", None):
                return teacher[name]
    return default


def _present(config: dict, runtime: dict, *names: str) -> bool:
    value = _field(config, runtime, *names, default=None)
    return value not in (None, "")


def validate_variant_config(config, dataset_meta, runtime_diagnostics) -> dict:
    """Return hard-gate validity for a SOTA row/config."""

    config = dict(config or {})
    dataset_meta = dict(dataset_meta or {})
    runtime = dict(runtime_diagnostics or {})
    merged = {**dataset_meta, **runtime, **config}
    variant = _variant_name(merged)
    reasons: list[str] = []
    warnings: list[str] = []

    if "coverage_medoid" in variant or str(merged.get("prototype_mode", "")).lower() == "coverage_medoid":
        warnings.append("coverage_medoid is deprecated in this sprint; use teacher_demand_herding instead.")

    if "sehgnn" in variant or "metapath" in variant:
        model_type = str(_field(merged, runtime, "model_type", "model", default="")).lower()
        documented = bool(_field(merged, runtime, "sehgnn_lite_equivalent_documented", "model_equivalent_documented", default=False))
        if model_type != "sehgnn_lite" and not documented:
            reasons.append("sehgnn_or_metapath_requires_model_type_sehgnn_lite")
        block_check = require_nonempty_feature_blocks(merged, required_prefix="metapath")
        reasons.extend(block_check["reasons"])
        feature_blocks = coerce_list(_field(merged, runtime, "feature_blocks", "compiled_blocks", default=[]))
        if not feature_blocks:
            reasons.append("feature_block_list_missing")
        if not _present(merged, runtime, "block_norm_stats_source", "compiled_block_stats_source"):
            reasons.append("block_norm_stats_source_missing")

    if "kd" in variant or bool(_field(merged, runtime, "use_kd", default=False)):
        teacher_type = str(_field(merged, runtime, "teacher_type", "type", default="none")).lower()
        if teacher_type in {"", "none"}:
            reasons.append("kd_teacher_type_missing")
        required = {
            "teacher_train_acc": ("teacher_train_acc", "train_acc"),
            "teacher_val_acc": ("teacher_val_acc", "val_acc"),
            "teacher_predicted_class_count": ("teacher_predicted_class_count", "predicted_class_count"),
            "kd_lambda": ("kd_lambda", "lambda_kd", "kd_weight"),
            "temperature": ("temperature", "kd_temperature"),
            "ce_loss": ("ce_loss", "ce_loss_end"),
            "kd_loss": ("kd_loss", "kd_loss_end"),
        }
        for reason_name, aliases in required.items():
            if not _present(merged, runtime, *aliases):
                reasons.append(f"{reason_name}_missing")

    if "path_lad" in variant or "pathlad" in variant:
        if not coerce_list(_field(merged, runtime, "path_lad_blocks", default=[])):
            reasons.append("path_lad_blocks_empty")
        if _field(merged, runtime, "path_lad_uses_train_labels_only", default=None) is not True:
            reasons.append("path_lad_train_label_only_not_logged")
        if not _present(merged, runtime, "path_lad_row_normalize", "path_lad_normalize", "path_lad_row_normalization"):
            reasons.append("path_lad_row_normalization_missing")
        if not _present(merged, runtime, "path_lad_leave_one_out", "path_lad_leave_one_out_for_train"):
            reasons.append("path_lad_leave_one_out_missing")
        if not _present(merged, runtime, "path_lad_hub_clip_quantile", "path_lad_hub_clip_thresholds"):
            reasons.append("path_lad_hub_clipping_missing")

    return {"valid": not reasons, "reasons": reasons, "warnings": warnings}


def assert_or_mark_invalid(row: dict, checks: dict) -> dict:
    """Attach invalid_config status instead of silently logging misleading rows."""

    out = dict(row)
    out["invalid_reasons"] = list(checks.get("reasons", []))
    out["warnings"] = list(checks.get("warnings", []))
    if not bool(checks.get("valid", False)):
        out["status"] = "invalid_config"
        out["accuracy"] = None
        out["macro_f1"] = None
    return out


def parse_metric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
