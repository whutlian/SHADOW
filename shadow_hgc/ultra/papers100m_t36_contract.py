from __future__ import annotations

from collections import Counter
from typing import Any

from shadow_hgc.ultra.papers100m_contract import PAPERS100M_NUM_NODES, truthy


DISCO_PARITY_RATIOS: tuple[float, ...] = (0.00005, 0.00010, 0.00020, 0.00050)
SCALE_FIDELITY_RATIOS: tuple[float, ...] = (0.00050, 0.00100, 0.00200, 0.00500, 0.01000)


T36_REQUIRED_FIELDS: list[str] = [
    "stage",
    "dataset",
    "method",
    "backend",
    "comparison_type",
    "seed",
    "requested_full_node_ratio",
    "ratio_percent",
    "full_node_denominator",
    "condensed_nodes",
    "target_universe_size",
    "target_universe_ratio",
    "cache_build_id",
    "edge_cache_id",
    "sft_cache_id",
    "teacher_cache_id",
    "selection_bank_id",
    "nested_bank_id",
    "teacher_id",
    "teacher_test_acc",
    "teacher_valid_acc",
    "teacher_cache_mode",
    "teacher_cache_bytes",
    "uses_streaming_logits",
    "uses_dense_teacher_cache_in_ram",
    "uses_dense_all_node_teacher_cache",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_exact_all_pair_distance",
    "uses_full_class_kmeans",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "edge_slice_cache_reused",
    "sft_cache_reused",
    "teacher_cache_reused",
    "selection_bank_reused",
    "incremental_edge_scans_after_cache_build",
    "hard_anchor_frac",
    "rare_repair_frac",
    "boundary_frac",
    "lambda_hard",
    "soft_temperature",
    "lambda_prior",
    "ant_enabled",
    "ant_edge_topk",
    "ant_link_predictor_id",
    "ant_edges",
    "ant_candidate_count",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "disco_acc",
    "random_acc_baseline",
    "herding_acc_baseline",
    "kcenter_acc_baseline",
    "absolute_gain_vs_disco",
    "relative_error_reduction_vs_disco",
    "beats_disco",
    "promotion_status",
    "failure_reason",
    "materialize_time",
    "student_train_time",
    "eval_time",
    "condensed_bytes",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "notes",
]


T36_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_dense_teacher_cache_in_ram",
    "uses_dense_all_node_teacher_cache",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_exact_all_pair_distance",
    "uses_full_class_kmeans",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)


def percent_to_ratio_decimal(percent: float | str) -> float:
    """Convert a percent value such as 0.005 into decimal ratio 0.00005."""
    return float(percent) / 100.0


def ratio_decimal_to_percent(ratio: float | str) -> float:
    return float(ratio) * 100.0


def ratio_matches(value: float, candidates: tuple[float, ...], *, atol: float = 5e-12) -> bool:
    return any(abs(float(value) - candidate) <= atol for candidate in candidates)


def comparison_type_for_backend(backend: str, ratio: float) -> str:
    if str(backend).lower() == "sgc" and ratio_matches(float(ratio), DISCO_PARITY_RATIOS):
        return "disco_parity"
    if ratio_matches(float(ratio), SCALE_FIDELITY_RATIOS):
        return "scale_fidelity"
    return "ours_native_not_disco_parity"


def make_t36_row(
    *,
    stage: str = "T36",
    dataset: str = "ogbn-papers100M",
    method: str = "",
    backend: str = "",
    comparison_type: str | None = None,
    seed: int = 7,
    requested_full_node_ratio: float = 0.0,
    full_node_denominator: int = PAPERS100M_NUM_NODES,
    condensed_nodes: int = 0,
    target_universe_size: int = 0,
    promotion_status: str = "diagnostic",
    failure_reason: str = "",
    **fields: Any,
) -> dict[str, Any]:
    ratio = float(requested_full_node_ratio)
    denom = int(full_node_denominator)
    target_size = int(target_universe_size)
    row: dict[str, Any] = {
        "stage": stage,
        "dataset": dataset,
        "method": method,
        "backend": backend,
        "comparison_type": comparison_type or comparison_type_for_backend(backend, ratio),
        "seed": int(seed),
        "requested_full_node_ratio": ratio,
        "ratio_percent": ratio_decimal_to_percent(ratio),
        "full_node_denominator": denom,
        "condensed_nodes": int(condensed_nodes),
        "target_universe_size": target_size,
        "target_universe_ratio": float(condensed_nodes) / float(target_size) if target_size else 0.0,
        "cache_build_id": "",
        "edge_cache_id": "",
        "sft_cache_id": "",
        "teacher_cache_id": "",
        "selection_bank_id": "",
        "nested_bank_id": "",
        "teacher_id": "",
        "teacher_test_acc": "",
        "teacher_valid_acc": "",
        "teacher_cache_mode": "",
        "teacher_cache_bytes": "",
        "uses_streaming_logits": True,
        "uses_dense_teacher_cache_in_ram": False,
        "uses_dense_all_node_teacher_cache": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_exact_all_pair_distance": False,
        "uses_full_class_kmeans": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "edge_slice_cache_reused": True,
        "sft_cache_reused": True,
        "teacher_cache_reused": True,
        "selection_bank_reused": True,
        "incremental_edge_scans_after_cache_build": 0,
        "hard_anchor_frac": "",
        "rare_repair_frac": "",
        "boundary_frac": "",
        "lambda_hard": "",
        "soft_temperature": "",
        "lambda_prior": "",
        "ant_enabled": False,
        "ant_edge_topk": 0,
        "ant_link_predictor_id": "",
        "ant_edges": 0,
        "ant_candidate_count": 0,
        "accuracy": "",
        "macro_f1": "",
        "valid_acc": "",
        "predicted_classes": "",
        "disco_acc": "",
        "random_acc_baseline": "",
        "herding_acc_baseline": "",
        "kcenter_acc_baseline": "",
        "absolute_gain_vs_disco": "",
        "relative_error_reduction_vs_disco": "",
        "beats_disco": "",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "materialize_time": "",
        "student_train_time": "",
        "eval_time": "",
        "condensed_bytes": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "notes": "",
    }
    row.update(fields)
    for field in T36_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t36_row(row: dict[str, Any]) -> dict[str, Any]:
    flags = [f"missing_field:{field}" for field in T36_REQUIRED_FIELDS if field not in row]
    promoted = str(row.get("promotion_status", "")).lower() == "promoted"
    if promoted:
        for flag in T36_FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        for flag in ("edge_slice_cache_reused", "sft_cache_reused", "teacher_cache_reused", "selection_bank_reused"):
            if not truthy(row.get(flag, False)):
                flags.append(f"{flag}_false")
        if int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) != 0:
            flags.append("incremental_edge_scans_after_cache_build_nonzero")
        backend = str(row.get("backend", "")).lower()
        ratio = float(row.get("requested_full_node_ratio", 0.0) or 0.0)
        if str(row.get("comparison_type", "")) == "disco_parity":
            if backend != "sgc":
                flags.append("disco_parity_backend_not_sgc")
            if not ratio_matches(ratio, DISCO_PARITY_RATIOS):
                flags.append("disco_parity_ratio_mismatch")
            if int(row.get("full_node_denominator", 0) or 0) != PAPERS100M_NUM_NODES:
                flags.append("disco_parity_denominator_mismatch")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def audit_one_cache_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        reasons.append("rows_missing")
    for key in ("edge_cache_id", "sft_cache_id", "teacher_cache_id"):
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
    return {"valid": not reasons, "failure_reasons": sorted(counts), "rows": len(rows), "reason_counts": dict(sorted(counts.items()))}


def summarize_stage_gates(
    *,
    teacher_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
    disco_rows: list[dict[str, Any]],
    ant_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    promoted_or_diag = disco_rows + scale_rows
    forbidden_hits = [validate_t36_row(row) for row in promoted_or_diag]
    teacher_gate = any(float(row.get("test_acc", row.get("accuracy", 0.0)) or 0.0) >= 0.55 for row in teacher_rows)
    nested_gate = bool(nested_rows) and all(int(row.get("prefix_violation_count", 0) or 0) == 0 for row in nested_rows)
    no_rebuild = audit_one_cache_reuse(promoted_or_diag)
    disco_005_rows = [row for row in disco_rows if abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 0.0005) < 1e-12]
    disco_002_rows = [row for row in disco_rows if abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 0.0002) < 1e-12]
    return {
        "S0_teacher_upgrade_rows_present": bool(teacher_rows),
        "S1_teacher_first_gate_0p55": teacher_gate,
        "S2_nested_bank_prefix_gate": nested_gate,
        "S3_disco_parity_rows_present": bool(disco_rows),
        "S4_disco_parity_sgc_backend_present": any(str(row.get("backend", "")).lower() == "sgc" for row in disco_rows),
        "S5_disco_0p05_maintain_gate": any(float(row.get("accuracy", 0.0) or 0.0) >= 0.510 for row in disco_005_rows),
        "S6_disco_0p02_first_gate": any(float(row.get("accuracy", 0.0) or 0.0) >= 0.45 for row in disco_002_rows),
        "S7_no_rebuild_gate": bool(no_rebuild.get("valid")),
        "S8_forbidden_guard_hits": sum(0 if item["valid"] else 1 for item in forbidden_hits),
        "S9_ant_boundedness_gate": bool(ant_rows) and all(truthy(row.get("ant_bounded", True)) for row in ant_rows),
        "S10_scale_fidelity_rows_present": bool(scale_rows),
        "row_count_teacher": len(teacher_rows),
        "row_count_nested": len(nested_rows),
        "row_count_disco": len(disco_rows),
        "row_count_ant": len(ant_rows),
        "row_count_scale": len(scale_rows),
        "no_rebuild_failure_reasons": ",".join(no_rebuild.get("failure_reasons", [])),
    }
