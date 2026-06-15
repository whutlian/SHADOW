from __future__ import annotations

from collections import Counter
from typing import Any

from shadow_hgc.sft.unified_auto_v2 import compute_t40_schedule, schedule_to_row_fields_v2
from shadow_hgc.sft.unified_stt import MAJORITY_VALID_ACC, NUM_CLASSES, NUM_NODES, fvalue, full_node_ratio, ivalue, truthy


PUBLIC_METHOD_ID = "shadow_stt_unified_auto_v2"
PUBLIC_METHOD_NAME = "Shadow-HGC-STT-U"

FIXED_CANDIDATE_POLICIES: tuple[str, ...] = (
    "auto_base",
    "coverage_heavy",
    "domain_coverage",
    "teacher_transport",
    "high_fidelity",
)

LEGACY_SPECIALIZED_METHOD_IDS: tuple[str, ...] = (
    "products_uca_hybrid_mixup",
    "reddit_ttcpp_gamlp_table_student",
    "reddit_stt_gamlp_ratio_v2",
    "scr_full_stochastic_coverage_plus_teacher_weight",
    "stt_randcore_sagn",
)

T40_MAIN_FIELDS: list[str] = [
    "dataset",
    "method",
    "public_method_name",
    "seed",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "ratio_mode",
    "condensed_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "valid_macro_f1",
    "budget_phase",
    "class_capacity_b",
    "teacher_reliability_q",
    "teacher_cache_policy",
    "teacher_cache_mode",
    "teacher_cache_k",
    "teacher_cache_bytes",
    "uses_dense_teacher_cache",
    "uses_dense_all_node_teacher_cache",
    "uses_teacher_probs_as_soft_targets",
    "uses_teacher_probs_as_input_features",
    "selected_policy",
    "policy_candidate_count",
    "policy_selection_score",
    "candidate_policy_scores_json",
    "coverage_weight",
    "hard_anchor_weight",
    "domain_weight",
    "soft_teacher_weight",
    "boundary_weight",
    "rare_weight",
    "mixup_weight",
    "selected_prior_kl",
    "domain_coverage_gap",
    "domain_gap_train_all",
    "coverage_bucket_count",
    "selected_class_count",
    "predicted_classes",
    "student_family",
    "student_internal_style",
    "student_capacity",
    "hidden_dim",
    "epochs",
    "soft_temperature",
    "lambda_hard",
    "lambda_soft",
    "lambda_prior",
    "lambda_domain",
    "lambda_mixup",
    "shared_cache_time_sec",
    "post_cache_time_sec",
    "selection_time_sec",
    "materialize_time_sec",
    "train_time_sec",
    "eval_time_sec",
    "storage",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "edge_cache_id",
    "sft_cache_id",
    "teacher_cache_id",
    "reservoir_cache_id",
    "cache_reused",
    "incremental_edge_scans_after_cache_build",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "table_role",
    "promotion_status",
    "failure_reason",
    "notes",
]

FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_teacher_probs_as_input_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
)


def _is_legacy_method(method: Any) -> bool:
    text = str(method)
    return text in LEGACY_SPECIALIZED_METHOD_IDS or text.startswith(("products_uca_", "reddit_ttcpp_", "reddit_stt_", "scr_", "stt_randcore_"))


def make_t40_row(
    *,
    dataset: str,
    requested_full_node_ratio: float,
    condensed_nodes: int | None = None,
    num_classes: int | None = None,
    method: str = PUBLIC_METHOD_ID,
    public_method_name: str = PUBLIC_METHOD_NAME,
    seed: int = 42,
    ratio_mode: str = "full_node",
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    valid_macro_f1: float | str = "",
    selected_policy: str = "auto_base",
    policy_candidate_count: int = len(FIXED_CANDIDATE_POLICIES),
    policy_selection_score: float | str = "",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    teacher_valid_acc: float | None = None,
    majority_valid_acc: float | None = None,
    domain_gap_train_all: float = 0.0,
    num_teacher_nodes: int | None = None,
    dense_cache_budget_bytes: int = 256 * 1024 * 1024,
    is_ultra_dataset: bool | None = None,
    table_role: str = "main",
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
    if is_ultra_dataset is None:
        is_ultra_dataset = canonical_dataset == "ogbn-papers100M"
    schedule = compute_t40_schedule(
        condensed_nodes=int(condensed_nodes),
        num_classes=classes,
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=majority_valid_acc,
        domain_gap_train_all=float(domain_gap_train_all),
        num_nodes=nodes,
        num_teacher_nodes=num_teacher_nodes if num_teacher_nodes is not None else fields.get("target_universe_size", nodes),
        is_ultra_dataset=bool(is_ultra_dataset),
        dense_cache_budget_bytes=int(dense_cache_budget_bytes),
    )
    row: dict[str, Any] = {
        "dataset": canonical_dataset,
        "method": method,
        "public_method_name": public_method_name,
        "seed": int(seed),
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": full_node_ratio(condensed_nodes=int(condensed_nodes), original_num_nodes=nodes),
        "ratio_mode": ratio_mode,
        "condensed_nodes": int(condensed_nodes),
        "condensed_edges": fields.get("condensed_edges", 0),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "valid_macro_f1": valid_macro_f1,
        "uses_teacher_probs_as_soft_targets": False,
        "uses_teacher_probs_as_input_features": False,
        "selected_policy": selected_policy,
        "policy_candidate_count": int(policy_candidate_count),
        "policy_selection_score": policy_selection_score,
        "candidate_policy_scores_json": "",
        "selected_prior_kl": fields.get("selected_prior_kl", 0.0),
        "domain_coverage_gap": fields.get("domain_coverage_gap", 0.0),
        "coverage_bucket_count": fields.get("coverage_bucket_count", 0),
        "selected_class_count": fields.get("selected_class_count", ""),
        "predicted_classes": fields.get("predicted_classes", ""),
        "shared_cache_time_sec": "",
        "post_cache_time_sec": "",
        "selection_time_sec": "",
        "materialize_time_sec": "",
        "train_time_sec": "",
        "eval_time_sec": "",
        "storage": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "edge_cache_id": "",
        "sft_cache_id": "",
        "teacher_cache_id": "",
        "reservoir_cache_id": "",
        "cache_reused": False,
        "incremental_edge_scans_after_cache_build": "",
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "table_role": table_role,
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": notes,
    }
    row.update(schedule_to_row_fields_v2(schedule))
    row.update(fields)
    row["selected_policy"] = selected_policy
    row["policy_candidate_count"] = int(policy_candidate_count)
    row["domain_gap_train_all"] = float(row.get("domain_gap_train_all", domain_gap_train_all) or 0.0)
    for field in T40_MAIN_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t40_main_row(row: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = [f"missing_field:{field}" for field in T40_MAIN_FIELDS if field not in row]
    if str(row.get("method", "")) != PUBLIC_METHOD_ID:
        flags.append("non_unified_method_id_in_main_table")
    if str(row.get("public_method_name", "")) != PUBLIC_METHOD_NAME:
        flags.append("public_method_name_mismatch")
    if _is_legacy_method(row.get("method", "")):
        flags.append("legacy_specialized_method_id_in_main_table")
    selected = str(row.get("selected_policy", ""))
    if selected not in FIXED_CANDIDATE_POLICIES:
        flags.append("selected_policy_not_in_fixed_candidate_set")
    if str(row.get("promotion_status", "")).lower() == "promoted":
        for flag in FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        if str(row.get("dataset", "")) == "ogbn-papers100M" and truthy(row.get("uses_dense_all_node_teacher_cache", False)):
            flags.append("uses_dense_all_node_teacher_cache")
        for metric in ("accuracy", "macro_f1", "valid_acc"):
            if row.get(metric) in {"", None}:
                flags.append(f"missing_{metric}")
    if truthy(row.get("uses_teacher_probs_as_input_features", False)) and truthy(row.get("uses_teacher_probs_as_soft_targets", False)):
        flags.append("teacher_probs_soft_target_and_input_conflict")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def audit_t40_papers100m_one_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
    papers = [
        row
        for row in rows
        if str(row.get("dataset")) == "ogbn-papers100M"
        and str(row.get("table_role", "main")) == "main"
        and str(row.get("promotion_status", "")).lower() == "promoted"
    ]
    flags: list[str] = []
    if not papers:
        return {"valid": True, "forbidden_flags": []}
    for key in ("edge_cache_id", "sft_cache_id", "teacher_cache_id", "reservoir_cache_id"):
        values = {str(row.get(key, "")) for row in papers if str(row.get(key, ""))}
        if not values:
            flags.append(f"papers100m_{key}_missing")
        if len(values) > 1:
            flags.append(f"papers100m_{key}_mismatch")
    for row in papers:
        if not truthy(row.get("cache_reused", False)):
            flags.append("papers100m_cache_reused_false")
        if ivalue(row.get("incremental_edge_scans_after_cache_build"), 0) != 0:
            flags.append("papers100m_incremental_edge_scans_after_cache_build_nonzero")
        if truthy(row.get("uses_dense_all_node_teacher_cache", False)):
            flags.append("papers100m_dense_all_node_teacher_cache")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags)), "reason_counts": dict(Counter(flags))}


def validate_t40_main_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags: list[str] = []
    for index, row in enumerate(rows):
        result = validate_t40_main_row(row)
        flags.extend(f"row_{index}:{flag}" for flag in result["forbidden_flags"] if flag.startswith("missing_field:"))
        flags.extend(flag for flag in result["forbidden_flags"] if not flag.startswith("missing_field:"))
    flags.extend(audit_t40_papers100m_one_cache(rows)["forbidden_flags"])
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}
