from __future__ import annotations

from collections import Counter
from typing import Any


T33_STAGE = "t33"

REDDIT_NUM_NODES = 232_965
ARXIV_NUM_NODES = 169_343
PRODUCTS_NUM_NODES = 2_449_029
PAPERS100M_NUM_NODES = 111_059_956
MAG240M_NUM_NODES = 121_751_666


T33_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "method",
    "stage",
    "seed",
    "status",
    "failure_reason",
    "promotion_status",
    "promotion_track",
    "promotion_allowed",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "total_condensed_nodes",
    "shadow_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "teacher_cache_mode",
    "teacher_cache_bytes",
    "teacher_ensemble_size",
    "teacher_accuracy",
    "teacher_valid_acc",
    "teacher_temperature",
    "teacher_entropy_mean",
    "teacher_disagreement_mean",
    "teacher_pairwise_kl_mean",
    "teacher_pairwise_kl_min",
    "teacher_pairwise_kl_max",
    "teacher_cache_duplicate_detected",
    "teacher_cache_hashes",
    "uses_teacher_logits",
    "uses_teacher_probs",
    "uses_logits_as_input",
    "uses_kd",
    "candidate_nodes_mode",
    "budget_policy",
    "core_frac_actual",
    "boundary_frac_actual",
    "disagreement_frac_actual",
    "rare_frac_actual",
    "anchor_frac_actual",
    "prior_repair_frac_actual",
    "selected_rows_per_class_min",
    "selected_rows_per_class_median",
    "selected_rows_per_class_max",
    "soft_class_mass_per_class_min",
    "soft_class_mass_per_class_median",
    "soft_class_mass_per_class_max",
    "selected_soft_prior_kl_to_teacher_prior",
    "selected_hard_anchor_count",
    "entropy_bucket_coverage",
    "margin_bucket_coverage",
    "degree_bucket_coverage",
    "signature_bucket_coverage",
    "virtual_mixup_enabled",
    "virtual_mixup_count",
    "student_model",
    "hidden_dim",
    "epochs",
    "soft_temperature",
    "lambda_soft",
    "lambda_hard",
    "lambda_prior",
    "lambda_mix",
    "checkpoint_selection",
    "base_predictor",
    "base_accuracy",
    "base_valid_acc",
    "base_macro_f1",
    "cns_accuracy",
    "cns_valid_acc",
    "cns_macro_f1",
    "graph_direction",
    "normalization_mode",
    "self_loop_mode",
    "correction_alpha",
    "smoothing_alpha",
    "correction_steps",
    "smoothing_steps",
    "autoscale",
    "feature_checksum",
    "mask_checksum",
    "edge_checksum",
    "logits_cache_path",
    "logits_cache_hash",
    "semantic_encoder",
    "semantic_cache_path",
    "semantic_dim",
    "uses_external_text_features",
    "precompute_time",
    "selection_time",
    "condensation_time",
    "training_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "sft_cache_bytes",
    "full_edge_scans",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_pair_distance",
    "uses_dense_teacher_cache_in_ram",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "planned_condensed_nodes",
    "teacher_topk_cache_bytes",
    "teacher_dense_cache_bytes_diagnostic",
    "selection_reservoir_bytes",
    "estimated_edge_scans",
    "estimated_selection_time",
    "estimated_peak_cpu_ram",
    "estimated_peak_gpu_ram",
    "notes",
    "next_action",
]


SAFE_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_teacher_logits",
    "uses_teacher_probs",
    "uses_logits_as_input",
    "uses_kd",
    "uses_external_text_features",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_pair_distance",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)

SOTA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_pair_distance",
)

ULTRA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_all_pair_distance",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
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


def original_num_nodes(dataset: str) -> int:
    if dataset == "Reddit":
        return REDDIT_NUM_NODES
    if dataset == "ogbn-arxiv":
        return ARXIV_NUM_NODES
    if dataset == "ogbn-products":
        return PRODUCTS_NUM_NODES
    if dataset == "ogbn-papers100M":
        return PAPERS100M_NUM_NODES
    if dataset == "MAG240M":
        return MAG240M_NUM_NODES
    raise ValueError(f"unknown T33 dataset: {dataset}")


def ratio_budget(dataset: str, requested_full_node_ratio: float) -> int:
    return max(1, int(round(original_num_nodes(dataset) * float(requested_full_node_ratio))))


def default_flags() -> dict[str, bool]:
    return {
        "uses_teacher_logits": False,
        "uses_teacher_probs": False,
        "uses_logits_as_input": False,
        "uses_kd": False,
        "uses_external_text_features": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_all_pair_distance": False,
        "uses_dense_teacher_cache_in_ram": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "virtual_mixup_enabled": False,
        "teacher_cache_duplicate_detected": False,
    }


def make_t33_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    total_condensed_nodes: int | None = None,
    shadow_nodes: int = 0,
    condensed_edges: int = 0,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    predicted_classes: int | str = "",
    status: str = "blocked",
    failure_reason: str = "",
    promotion_track: str = "safe_main",
    promotion_status: str = "not_promoted",
    ratio_mode: str = "full_node_ratio",
    notes: str = "",
    next_action: str = "",
    extra: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    nodes = original_num_nodes(dataset)
    if total_condensed_nodes is None:
        total_condensed_nodes = ratio_budget(dataset, requested_full_node_ratio) if requested_full_node_ratio else 0
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "stage": T33_STAGE,
        "seed": int(seed),
        "status": status,
        "failure_reason": failure_reason,
        "promotion_status": promotion_status,
        "promotion_track": promotion_track,
        "promotion_allowed": promotion_status == "promoted",
        "ratio_mode": ratio_mode,
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(total_condensed_nodes) / float(nodes) if nodes else 0.0,
        "original_num_nodes": nodes,
        "total_condensed_nodes": int(total_condensed_nodes),
        "shadow_nodes": int(shadow_nodes),
        "condensed_edges": int(condensed_edges),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "teacher_cache_mode": "",
        "teacher_cache_bytes": "",
        "teacher_ensemble_size": "",
        "teacher_accuracy": "",
        "teacher_valid_acc": "",
        "teacher_temperature": "",
        "teacher_entropy_mean": "",
        "teacher_disagreement_mean": "",
        "teacher_pairwise_kl_mean": "",
        "teacher_pairwise_kl_min": "",
        "teacher_pairwise_kl_max": "",
        "teacher_cache_hashes": "",
        "candidate_nodes_mode": "",
        "budget_policy": "",
        "core_frac_actual": "",
        "boundary_frac_actual": "",
        "disagreement_frac_actual": "",
        "rare_frac_actual": "",
        "anchor_frac_actual": "",
        "prior_repair_frac_actual": "",
        "selected_rows_per_class_min": "",
        "selected_rows_per_class_median": "",
        "selected_rows_per_class_max": "",
        "soft_class_mass_per_class_min": "",
        "soft_class_mass_per_class_median": "",
        "soft_class_mass_per_class_max": "",
        "selected_soft_prior_kl_to_teacher_prior": "",
        "selected_hard_anchor_count": "",
        "entropy_bucket_coverage": "",
        "margin_bucket_coverage": "",
        "degree_bucket_coverage": "",
        "signature_bucket_coverage": "",
        "virtual_mixup_count": "",
        "student_model": "",
        "hidden_dim": "",
        "epochs": "",
        "soft_temperature": "",
        "lambda_soft": "",
        "lambda_hard": "",
        "lambda_prior": "",
        "lambda_mix": "",
        "checkpoint_selection": "",
        "base_predictor": "",
        "base_accuracy": "",
        "base_valid_acc": "",
        "base_macro_f1": "",
        "cns_accuracy": "",
        "cns_valid_acc": "",
        "cns_macro_f1": "",
        "graph_direction": "",
        "normalization_mode": "",
        "self_loop_mode": "",
        "correction_alpha": "",
        "smoothing_alpha": "",
        "correction_steps": "",
        "smoothing_steps": "",
        "autoscale": "",
        "feature_checksum": "",
        "mask_checksum": "",
        "edge_checksum": "",
        "logits_cache_path": "",
        "logits_cache_hash": "",
        "semantic_encoder": "",
        "semantic_cache_path": "",
        "semantic_dim": "",
        "precompute_time": "",
        "selection_time": "",
        "condensation_time": "",
        "training_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "sft_cache_bytes": "",
        "full_edge_scans": "",
        "planned_condensed_nodes": "",
        "teacher_topk_cache_bytes": "",
        "teacher_dense_cache_bytes_diagnostic": "",
        "selection_reservoir_bytes": "",
        "estimated_edge_scans": "",
        "estimated_selection_time": "",
        "estimated_peak_cpu_ram": "",
        "estimated_peak_gpu_ram": "",
        "notes": notes,
        "next_action": next_action,
        **default_flags(),
    }
    row.update(fields)
    if extra:
        row.update(extra)
    for field in T33_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def reddit_gate_status(*, ratio: float, accuracy: float, macro_f1: float) -> tuple[str, str]:
    ratio = float(ratio)
    acc = float(accuracy)
    macro = float(macro_f1)
    if abs(ratio - 0.0005) < 1e-12:
        return ("promoted", "") if acc >= 0.900 else ("not_promoted", "ttcpp_accuracy_gate_not_met")
    if abs(ratio - 0.001) < 1e-12:
        if acc < 0.923:
            return "not_promoted", "ttcpp_accuracy_gate_not_met"
        if macro < 0.885:
            return "not_promoted", "ttcpp_macro_regression"
        return "promoted", ""
    if abs(ratio - 0.002) < 1e-12:
        return ("promoted", "") if acc >= 0.932 else ("not_promoted", "ttcpp_accuracy_gate_not_met")
    if abs(ratio - 0.0025) < 1e-12:
        return ("promoted", "") if acc >= 0.934 else ("not_promoted", "ttcpp_accuracy_gate_not_met")
    if abs(ratio - 0.005) < 1e-12:
        if acc < 0.938:
            return "not_promoted", "ttcpp_accuracy_gate_not_met"
        if macro < 0.904:
            return "not_promoted", "ttcpp_macro_regression"
        return "promoted", ""
    if abs(ratio - 0.01) < 1e-12:
        return ("promoted", "") if acc >= 0.940 else ("not_promoted", "ttcpp_accuracy_gate_not_met")
    return "not_promoted", "ttcpp_ratio_gate_not_defined"


def validate_t33_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = [f"missing_field:{field}" for field in T33_REQUIRED_FIELDS if field not in row]
    track = str(row.get("promotion_track", "safe_main"))
    if track == "ultra_planner":
        flags = ULTRA_FORBIDDEN_FLAGS
    elif track == "sota_chase":
        flags = SOTA_FORBIDDEN_FLAGS
    else:
        flags = SAFE_FORBIDDEN_FLAGS
    for flag in flags:
        if truthy(row.get(flag, False)):
            forbidden.append(flag)
    if track == "sota_chase" and wants_promotion(row):
        if str(row.get("teacher_cache_mode", "")) == "":
            forbidden.append("missing_teacher_cache_mode")
        if str(row.get("teacher_cache_bytes", "")) == "":
            forbidden.append("missing_teacher_cache_bytes")
    if track == "ultra_planner":
        if str(row.get("teacher_cache_mode", "")) == "dense_fp16":
            forbidden.append("ultra_dense_teacher_cache")
        if truthy(row.get("uses_dense_teacher_cache_in_ram", False)):
            forbidden.append("ultra_dense_teacher_cache")
    if wants_promotion(row):
        if row.get("accuracy") in {"", None} and track != "ultra_planner":
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None} and track != "ultra_planner":
            forbidden.append("missing_macro_f1")
        status = str(row.get("status", ""))
        if status in {"blocked", "failed"} or "smoke" in status or "carried_forward" in status:
            forbidden.append("status_not_promotable")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t33_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = guarded.get("promotion_status") or "not_promoted"
        return guarded
    result = validate_t33_row(guarded)
    if not result["valid"]:
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["promotion_allowed"] = False
        guarded["failure_reason"] = ",".join(sorted(set(result["forbidden_flags"])))
        return guarded
    guarded["promotion_status"] = "promoted"
    guarded["promotion_allowed"] = True
    return guarded


def summarize_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if wants_promotion(row)]
    unsafe = [row for row in promoted if not validate_t33_row(row)["valid"]]
    blocked = Counter(str(row.get("failure_reason", "")) for row in rows if str(row.get("failure_reason", "")))
    return {
        "rows": len(rows),
        "unsafe_promoted_rows": len(unsafe),
        "promoted_safe_rows": sum(1 for row in promoted if row.get("promotion_track") == "safe_main"),
        "promoted_sota_chase_rows": sum(1 for row in promoted if row.get("promotion_track") == "sota_chase"),
        "promoted_ultra_rows": sum(1 for row in promoted if row.get("promotion_track") == "ultra_planner"),
        "blocked_rows_by_reason": dict(sorted(blocked.items())),
    }
