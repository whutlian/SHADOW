from __future__ import annotations

from collections import Counter
from typing import Any

from shadow_hgc.sft.t40_contract import (
    FORBIDDEN_PROMOTED_FLAGS,
    LEGACY_SPECIALIZED_METHOD_IDS,
    T40_MAIN_FIELDS,
    _is_legacy_method,
)
from shadow_hgc.sft.unified_auto_v3 import (
    FIXED_CANDIDATE_POLICIES_V3,
    compute_t41_schedule,
    apply_candidate_policy,
    policy_selection_score_v2_equivalent,
    policy_selection_score_v3,
    schedule_to_row_fields_v3,
)
from shadow_hgc.sft.unified_stt import MAJORITY_VALID_ACC, NUM_CLASSES, NUM_NODES, full_node_ratio, ivalue, truthy


PUBLIC_METHOD_ID = "shadow_stt_unified_auto_v3"
PUBLIC_METHOD_NAME = "Shadow-HGC-STT-U"
FIXED_CANDIDATE_POLICIES: tuple[str, ...] = FIXED_CANDIDATE_POLICIES_V3

T41_NEW_FIELDS: list[str] = [
    "method_id",
    "method_name",
    "ratio",
    "micro_f1",
    "storage_bytes",
    "domain_transport_active",
    "domain_transport_strength",
    "domain_transport_rows",
    "domain_row_frac",
    "domain_gap_before",
    "domain_gap_after",
    "domain_transport_gain",
    "domain_overfit_proxy",
    "score_v2_equivalent",
    "score_v3",
    "row_type_counts",
]

T41_MAIN_FIELDS: list[str] = []
for _field in T40_MAIN_FIELDS + T41_NEW_FIELDS:
    if _field not in T41_MAIN_FIELDS:
        T41_MAIN_FIELDS.append(_field)


def make_t41_row(
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
    if majority_valid_acc is None:
        majority_valid_acc = MAJORITY_VALID_ACC.get(canonical_dataset)
    if is_ultra_dataset is None:
        is_ultra_dataset = canonical_dataset == "ogbn-papers100M"
    schedule = compute_t41_schedule(
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
    schedule = apply_candidate_policy(schedule, str(selected_policy))
    domain_rows = int(fields.get("domain_transport_rows", 0) or 0)
    domain_row_frac = float(fields.get("domain_row_frac", (domain_rows / max(1, int(condensed_nodes))) if domain_rows else 0.0) or 0.0)
    gap_before = float(fields.get("domain_gap_before", fields.get("domain_coverage_gap", 0.0)) or 0.0)
    gap_after = float(fields.get("domain_gap_after", gap_before) or 0.0)
    transport_gain = max(0.0, gap_before - gap_after)
    overfit_proxy = abs(gap_after - float(domain_gap_train_all))
    row_type_counts = fields.get("row_type_counts", "")
    if not row_type_counts:
        row_type_counts = f'{{"domain_transport":{domain_rows},"hard_anchor":{max(0, int(condensed_nodes) - domain_rows)}}}'
    row: dict[str, Any] = {
        "dataset": canonical_dataset,
        "method": method,
        "method_id": method,
        "public_method_name": public_method_name,
        "method_name": public_method_name,
        "seed": int(seed),
        "ratio": float(requested_full_node_ratio),
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": full_node_ratio(condensed_nodes=int(condensed_nodes), original_num_nodes=nodes),
        "ratio_mode": ratio_mode,
        "condensed_nodes": int(condensed_nodes),
        "condensed_edges": fields.get("condensed_edges", 0),
        "accuracy": accuracy,
        "micro_f1": fields.get("micro_f1", accuracy),
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
        "domain_coverage_gap": fields.get("domain_coverage_gap", gap_after),
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
        "storage_bytes": "",
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
        "domain_transport_rows": domain_rows,
        "domain_row_frac": domain_row_frac,
        "domain_gap_before": gap_before,
        "domain_gap_after": gap_after,
        "domain_transport_gain": transport_gain,
        "domain_overfit_proxy": overfit_proxy,
        "row_type_counts": row_type_counts,
    }
    row.update(schedule_to_row_fields_v3(schedule))
    row.update(fields)
    row["method"] = method
    row["method_id"] = method
    row["public_method_name"] = public_method_name
    row["method_name"] = public_method_name
    row["ratio"] = row.get("requested_full_node_ratio", float(requested_full_node_ratio))
    if row.get("micro_f1", "") in {"", None}:
        row["micro_f1"] = row.get("accuracy", "")
    if row.get("storage_bytes", "") in {"", None}:
        row["storage_bytes"] = row.get("storage", "")
    row["selected_policy"] = selected_policy
    row["policy_candidate_count"] = int(policy_candidate_count)
    row["domain_transport_rows"] = domain_rows
    row["domain_row_frac"] = domain_row_frac
    row["domain_gap_before"] = gap_before
    row["domain_gap_after"] = gap_after
    row["domain_transport_gain"] = transport_gain
    row["domain_overfit_proxy"] = overfit_proxy
    row["row_type_counts"] = row_type_counts
    row["domain_transport_active"] = bool(domain_rows > 0) if domain_rows else bool(row.get("domain_transport_active", False) and str(selected_policy) == "domain_transport")
    row["score_v2_equivalent"] = policy_selection_score_v2_equivalent(row) if valid_acc not in {"", None} else ""
    row["score_v3"] = policy_selection_score_v3(row) if valid_acc not in {"", None} else ""
    row["policy_selection_score"] = row["score_v3"] if policy_selection_score in {"", None} else policy_selection_score
    for field in T41_MAIN_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t41_main_row(row: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = [f"missing_field:{field}" for field in T41_MAIN_FIELDS if field not in row]
    if str(row.get("method", "")) != PUBLIC_METHOD_ID:
        flags.append("non_unified_method_id_in_main_table")
    if str(row.get("method_id", row.get("method", ""))) != PUBLIC_METHOD_ID:
        flags.append("non_unified_method_id_alias_in_main_table")
    if str(row.get("public_method_name", "")) != PUBLIC_METHOD_NAME:
        flags.append("public_method_name_mismatch")
    if str(row.get("method_name", row.get("public_method_name", ""))) != PUBLIC_METHOD_NAME:
        flags.append("public_method_name_alias_mismatch")
    if _is_legacy_method(row.get("method", "")):
        flags.append("legacy_specialized_method_id_in_main_table")
    selected = str(row.get("selected_policy", ""))
    if selected not in FIXED_CANDIDATE_POLICIES:
        flags.append("selected_policy_not_in_fixed_candidate_set")
    if str(row.get("promotion_status", "")).lower() == "promoted":
        for flag in FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        for metric in ("accuracy", "macro_f1", "valid_acc"):
            if row.get(metric) in {"", None}:
                flags.append(f"missing_{metric}")
        if str(row.get("dataset", "")) == "ogbn-papers100M":
            if truthy(row.get("uses_dense_all_node_teacher_cache", False)):
                flags.append("uses_dense_all_node_teacher_cache")
            if str(row.get("teacher_cache_mode", "")) not in {"topk8_tail", "topk16_tail"}:
                flags.append("papers100m_teacher_cache_mode_not_topk_tail")
    if truthy(row.get("uses_teacher_probs_as_input_features", False)) and truthy(row.get("uses_teacher_probs_as_soft_targets", False)):
        flags.append("teacher_probs_soft_target_and_input_conflict")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def audit_t41_papers100m_one_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        if str(row.get("teacher_cache_mode", "")) not in {"topk8_tail", "topk16_tail"}:
            flags.append("papers100m_teacher_cache_mode_not_topk_tail")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags)), "reason_counts": dict(Counter(flags))}


def validate_t41_main_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags: list[str] = []
    for index, row in enumerate(rows):
        result = validate_t41_main_row(row)
        flags.extend(f"row_{index}:{flag}" for flag in result["forbidden_flags"] if flag.startswith("missing_field:"))
        flags.extend(flag for flag in result["forbidden_flags"] if not flag.startswith("missing_field:"))
        if _is_legacy_method(row.get("method", "")) or str(row.get("method", "")) in LEGACY_SPECIALIZED_METHOD_IDS:
            flags.append("legacy_specialized_method_id_in_main_table")
    flags.extend(audit_t41_papers100m_one_cache(rows)["forbidden_flags"])
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}
