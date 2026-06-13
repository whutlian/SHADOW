from __future__ import annotations

from collections import Counter
from typing import Any


PAPERS100M_NUM_NODES = 111_059_956
PAPERS100M_NUM_EDGES = 1_615_685_872
PAPERS100M_NUM_CLASSES = 172
PAPERS100M_FEATURE_DIM = 128


T35_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "method",
    "seed",
    "status",
    "promotion_status",
    "failure_reason",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "full_node_ratio_denominator",
    "target_universe_size",
    "target_universe_ratio",
    "condensed_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "cache_root",
    "cache_build_id",
    "edge_slice_cache_id",
    "sft_cache_id",
    "teacher_cache_id",
    "selection_bank_id",
    "cache_reused",
    "edge_slice_cache_reused",
    "sft_cache_reused",
    "teacher_cache_reused",
    "selection_bank_reused",
    "incremental_edge_scans_after_cache_build",
    "num_nodes",
    "num_edges",
    "num_classes",
    "feature_dim",
    "train_size",
    "valid_size",
    "test_size",
    "sft_block_manifest",
    "teacher_cache_scope",
    "teacher_cache_mode",
    "teacher_cache_bytes",
    "teacher_dense_cache_bytes_diagnostic",
    "uses_dense_teacher_cache_in_ram",
    "uses_teacher_probs_as_input",
    "uses_teacher_probs_as_soft_targets",
    "selection_policy",
    "nested_selection",
    "bucket_core_count",
    "bucket_boundary_count",
    "bucket_rare_count",
    "bucket_prior_repair_count",
    "bucket_hard_anchor_count",
    "selected_soft_prior_kl",
    "teacher_target_prior_entropy",
    "precompute_time",
    "edge_cache_time",
    "sft_cache_time",
    "teacher_train_time",
    "teacher_infer_time",
    "selection_bank_time",
    "condensed_materialize_time",
    "student_train_time",
    "eval_time",
    "edge_cache_bytes",
    "sft_cache_bytes",
    "selection_bank_bytes",
    "condensed_cache_bytes",
    "total_cache_bytes",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_all_pair_distance",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "notes",
]


FORBIDDEN_PROMOTED_FLAGS = (
    "uses_dense_teacher_cache_in_ram",
    "uses_teacher_probs_as_input",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_all_pair_distance",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def make_t35_row(
    *,
    dataset: str = "ogbn-papers100M",
    method: str = "papers100m_stt_one_cache",
    seed: int = 42,
    requested_full_node_ratio: float = 0.0,
    condensed_nodes: int = 0,
    full_node_ratio_denominator: int = PAPERS100M_NUM_NODES,
    target_universe_size: int = 0,
    status: str = "completed",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    **fields: Any,
) -> dict[str, Any]:
    denominator = int(full_node_ratio_denominator) if int(full_node_ratio_denominator) > 0 else PAPERS100M_NUM_NODES
    target_size = int(target_universe_size)
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "status": status,
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(condensed_nodes) / float(denominator) if denominator else 0.0,
        "full_node_ratio_denominator": denominator,
        "target_universe_size": target_size,
        "target_universe_ratio": float(condensed_nodes) / float(target_size) if target_size else 0.0,
        "condensed_nodes": int(condensed_nodes),
        "condensed_edges": 0,
        "accuracy": "",
        "macro_f1": "",
        "valid_acc": "",
        "predicted_classes": "",
        "cache_root": "",
        "cache_build_id": "",
        "edge_slice_cache_id": "",
        "sft_cache_id": "",
        "teacher_cache_id": "",
        "selection_bank_id": "",
        "cache_reused": True,
        "edge_slice_cache_reused": True,
        "sft_cache_reused": True,
        "teacher_cache_reused": True,
        "selection_bank_reused": True,
        "incremental_edge_scans_after_cache_build": 0,
        "num_nodes": denominator,
        "num_edges": PAPERS100M_NUM_EDGES,
        "num_classes": PAPERS100M_NUM_CLASSES,
        "feature_dim": PAPERS100M_FEATURE_DIM,
        "train_size": "",
        "valid_size": "",
        "test_size": "",
        "sft_block_manifest": "",
        "teacher_cache_scope": "target_universe",
        "teacher_cache_mode": "topk8_tail",
        "teacher_cache_bytes": "",
        "teacher_dense_cache_bytes_diagnostic": "",
        "uses_dense_teacher_cache_in_ram": False,
        "uses_teacher_probs_as_input": False,
        "uses_teacher_probs_as_soft_targets": True,
        "selection_policy": "stt_ratio_v2",
        "nested_selection": True,
        "bucket_core_count": 0,
        "bucket_boundary_count": 0,
        "bucket_rare_count": 0,
        "bucket_prior_repair_count": 0,
        "bucket_hard_anchor_count": 0,
        "selected_soft_prior_kl": "",
        "teacher_target_prior_entropy": "",
        "precompute_time": "",
        "edge_cache_time": "",
        "sft_cache_time": "",
        "teacher_train_time": "",
        "teacher_infer_time": "",
        "selection_bank_time": "",
        "condensed_materialize_time": "",
        "student_train_time": "",
        "eval_time": "",
        "edge_cache_bytes": "",
        "sft_cache_bytes": "",
        "selection_bank_bytes": "",
        "condensed_cache_bytes": "",
        "total_cache_bytes": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_all_pair_distance": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "notes": "",
    }
    row.update(fields)
    for field in T35_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t35_row(row: dict[str, Any]) -> dict[str, Any]:
    flags = [f"missing_field:{field}" for field in T35_REQUIRED_FIELDS if field not in row]
    promoted = str(row.get("promotion_status", "")) == "promoted"
    if promoted:
        for flag in FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        if str(row.get("teacher_cache_scope", "")) != "target_universe":
            flags.append("teacher_cache_scope_not_target_universe")
        if str(row.get("teacher_cache_mode", "")).startswith("dense"):
            flags.append("dense_teacher_cache_mode_not_promotable")
        for field in ("edge_slice_cache_reused", "sft_cache_reused", "teacher_cache_reused", "selection_bank_reused"):
            if not truthy(row.get(field, False)):
                flags.append(f"{field}_false")
        if int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) != 0:
            flags.append("incremental_edge_scans_after_cache_build_nonzero")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def audit_cache_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        reasons.append("ratio_rows_missing")
    for key in ("edge_slice_cache_id", "sft_cache_id", "teacher_cache_id", "selection_bank_id"):
        values = {str(row.get(key, "")) for row in rows if str(row.get(key, ""))}
        if len(values) > 1:
            reasons.append(f"{key}_mismatch")
        if not values and rows:
            reasons.append(f"{key}_missing")
    for row in rows:
        if int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) != 0:
            reasons.append("incremental_edge_scans_after_cache_build_nonzero")
        for flag in ("edge_slice_cache_reused", "sft_cache_reused", "teacher_cache_reused", "selection_bank_reused"):
            if not truthy(row.get(flag, False)):
                reasons.append(f"{flag}_false")
    counts = Counter(reasons)
    return {
        "valid": not reasons,
        "failure_reasons": sorted(counts),
        "rows": len(rows),
        "reason_counts": dict(sorted(counts.items())),
    }
