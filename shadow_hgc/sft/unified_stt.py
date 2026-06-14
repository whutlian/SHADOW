from __future__ import annotations

from collections import Counter
from typing import Any

from shadow_hgc.sft.unified_schedule import compute_unified_schedule, full_node_ratio, schedule_to_row_fields


PUBLIC_METHOD_ID = "shadow_stt_unified_auto"
PUBLIC_METHOD_NAME = "Shadow-HGC-STT-U"
PUBLIC_TABLE_LABEL = "Shadow-HGC-STT-U (ours)"

OLD_SPECIALIZED_METHOD_PREFIXES: tuple[str, ...] = (
    "products_uca_",
    "reddit_ttcpp_",
    "reddit_stt_",
    "scr_",
    "stt_randcore_",
)

NUM_NODES: dict[str, int] = {
    "Reddit": 232_965,
    "ogbn-products": 2_449_029,
    "ogbn-arxiv": 169_343,
    "ogbn-papers100M": 111_059_956,
}

NUM_CLASSES: dict[str, int] = {
    "Reddit": 41,
    "ogbn-products": 47,
    "ogbn-arxiv": 40,
    "ogbn-papers100M": 172,
}

MAJORITY_VALID_ACC: dict[str, float] = {
    "Reddit": 0.310,
    "ogbn-products": 0.095,
    "ogbn-arxiv": 0.140,
    "ogbn-papers100M": 0.150,
}

T38_MAIN_FIELDS: list[str] = [
    "dataset",
    "method",
    "public_method",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "ratio_mode",
    "condensed_nodes",
    "backend",
    "comparison_type",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "promotion_status",
    "failure_reason",
    "budget_per_class",
    "budget_phase_tau",
    "teacher_reliability_q",
    "teacher_cache_policy",
    "teacher_cache_mode",
    "teacher_cache_bytes",
    "coverage_weight",
    "hard_weight",
    "soft_weight",
    "boundary_weight",
    "rare_weight",
    "diversity_weight",
    "alpha_hard",
    "alpha_soft",
    "alpha_prior",
    "alpha_mix",
    "soft_temperature",
    "student_family",
    "student_internal_style",
    "hidden_dim",
    "epochs",
    "shared_cache_time_sec",
    "post_cache_time_sec",
    "total_storage_bytes",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "edge_cache_id",
    "sft_cache_id",
    "teacher_cache_id",
    "unified_reservoir_id",
    "cache_reused",
    "incremental_edge_scans_after_cache_build",
    "uses_teacher_probs_as_soft_targets",
    "uses_teacher_probs_as_input_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_all_node_teacher_cache",
    "notes",
]

FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_teacher_probs_as_input_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_all_node_teacher_cache",
)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def fvalue(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ivalue(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_old_specialized_method(method: Any) -> bool:
    text = str(method)
    return any(text.startswith(prefix) for prefix in OLD_SPECIALIZED_METHOD_PREFIXES)


def make_t38_row(
    *,
    dataset: str,
    requested_full_node_ratio: float,
    condensed_nodes: int | None = None,
    num_classes: int | None = None,
    method: str = PUBLIC_METHOD_ID,
    public_method: str = PUBLIC_METHOD_NAME,
    backend: str = "stt_gated_mixer",
    comparison_type: str = "ours_native",
    ratio_mode: str = "full_node",
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    predicted_classes: int | str = "",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    teacher_valid_acc: float | None = None,
    majority_valid_acc: float | None = None,
    num_teacher_nodes: int | None = None,
    notes: str = "",
    **fields: Any,
) -> dict[str, Any]:
    canonical_dataset = str(dataset)
    nodes = NUM_NODES.get(canonical_dataset, ivalue(fields.get("num_nodes"), 0))
    classes = int(num_classes if num_classes is not None else NUM_CLASSES.get(canonical_dataset, 1))
    if condensed_nodes is None:
        condensed_nodes = max(1, int(round(float(requested_full_node_ratio) * float(nodes)))) if nodes else 0
    if teacher_valid_acc is None and valid_acc not in {"", None} and truthy(fields.get("uses_teacher_probs_as_soft_targets", False)):
        teacher_valid_acc = fvalue(valid_acc)
    if majority_valid_acc is None:
        majority_valid_acc = MAJORITY_VALID_ACC.get(canonical_dataset)
    schedule = compute_unified_schedule(
        condensed_nodes=int(condensed_nodes),
        num_classes=classes,
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=majority_valid_acc,
        num_nodes=nodes,
        num_teacher_nodes=num_teacher_nodes if num_teacher_nodes is not None else fields.get("target_universe_size", nodes),
    )
    row: dict[str, Any] = {
        "dataset": canonical_dataset,
        "method": method,
        "public_method": public_method,
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": full_node_ratio(condensed_nodes=int(condensed_nodes), original_num_nodes=nodes),
        "ratio_mode": ratio_mode,
        "condensed_nodes": int(condensed_nodes),
        "backend": backend,
        "comparison_type": comparison_type,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "shared_cache_time_sec": "",
        "post_cache_time_sec": "",
        "total_storage_bytes": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "edge_cache_id": "",
        "sft_cache_id": "",
        "teacher_cache_id": "",
        "unified_reservoir_id": "",
        "cache_reused": False,
        "incremental_edge_scans_after_cache_build": "",
        "uses_teacher_probs_as_soft_targets": False,
        "uses_teacher_probs_as_input_features": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "notes": notes,
    }
    row.update(schedule_to_row_fields(schedule))
    row.update(fields)
    for field in T38_MAIN_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t38_main_row(row: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = [f"missing_field:{field}" for field in T38_MAIN_FIELDS if field not in row]
    if str(row.get("method", "")) != PUBLIC_METHOD_ID:
        flags.append("main_method_id_mismatch")
    if str(row.get("public_method", "")) != PUBLIC_METHOD_NAME:
        flags.append("public_method_mismatch")
    if is_old_specialized_method(row.get("method", "")):
        flags.append("old_specialized_method_id_in_main")
    if str(row.get("promotion_status", "")).lower() == "promoted":
        for flag in FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        if row.get("accuracy") in {"", None}:
            flags.append("missing_accuracy")
        if row.get("macro_f1") in {"", None}:
            flags.append("missing_macro_f1")
    if truthy(row.get("uses_teacher_probs_as_input_features", False)) and truthy(row.get("uses_teacher_probs_as_soft_targets", False)):
        flags.append("teacher_probs_soft_target_and_input_conflict")
    if fvalue(row.get("teacher_reliability_q")) <= 0.0:
        if fvalue(row.get("alpha_soft")) != 0.0:
            flags.append("alpha_soft_nonzero_without_trusted_teacher")
        if fvalue(row.get("soft_weight")) != 0.0:
            flags.append("soft_weight_nonzero_without_trusted_teacher")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def audit_papers100m_one_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
    papers_rows = [row for row in rows if str(row.get("dataset")) == "ogbn-papers100M"]
    reasons: list[str] = []
    if not papers_rows:
        return {"valid": True, "forbidden_flags": []}
    for key in ("edge_cache_id", "sft_cache_id", "teacher_cache_id", "unified_reservoir_id"):
        values = {str(row.get(key, "")) for row in papers_rows if str(row.get(key, ""))}
        if not values:
            reasons.append(f"papers100m_{key}_missing")
        if len(values) > 1:
            reasons.append(f"papers100m_{key}_mismatch")
    for row in papers_rows:
        if not truthy(row.get("cache_reused", False)):
            reasons.append("papers100m_cache_reused_false")
        if ivalue(row.get("incremental_edge_scans_after_cache_build"), 0) != 0:
            reasons.append("papers100m_incremental_edge_scans_after_cache_build_nonzero")
    return {"valid": not reasons, "forbidden_flags": sorted(set(reasons)), "reason_counts": dict(Counter(reasons))}


def validate_t38_main_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags: list[str] = []
    for index, row in enumerate(rows):
        result = validate_t38_main_row(row)
        flags.extend(f"row_{index}:{flag}" for flag in result["forbidden_flags"] if flag.startswith("missing_field:"))
        flags.extend(flag for flag in result["forbidden_flags"] if not flag.startswith("missing_field:"))
    one_cache = audit_papers100m_one_cache(rows)
    flags.extend(one_cache["forbidden_flags"])
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def acceptable_gap(dataset: str) -> tuple[float, float]:
    if str(dataset) == "Reddit":
        return 0.003, 0.005
    if str(dataset) == "ogbn-products":
        return 0.010, 0.015
    if str(dataset) == "ogbn-papers100M":
        return 0.010, 0.010
    return 0.0, 0.0
