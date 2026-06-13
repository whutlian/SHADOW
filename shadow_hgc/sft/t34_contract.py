from __future__ import annotations

from collections import Counter
from typing import Any


T34_STAGE = "t34"

NUM_NODES = {
    "Reddit": 232_965,
    "ogbn-products": 2_449_029,
    "ogbn-arxiv": 169_343,
    "ogbn-papers100M": 111_059_956,
    "MAG240M": 121_751_666,
}


T34_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "num_nodes",
    "num_edges",
    "num_classes",
    "method",
    "stage",
    "seed",
    "status",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "condensed_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "valid_macro_f1",
    "valid_test_gap",
    "predicted_classes",
    "promotion_track",
    "promotion_status",
    "promotion_allowed",
    "failure_reason",
    "teacher_method",
    "teacher_accuracy",
    "teacher_valid_acc",
    "teacher_cache_mode",
    "teacher_cache_bytes",
    "teacher_dense_cache_bytes_diagnostic",
    "teacher_topk_cache_bytes",
    "cache_compression_ratio",
    "uses_teacher_probs",
    "uses_teacher_logits",
    "uses_logits_as_input",
    "uses_teacher_probs_as_input",
    "soft_target_only",
    "uses_external_text_features",
    "semantic_encoder",
    "semantic_cache_path",
    "semantic_cache_bytes",
    "semantic_features_are_frozen",
    "lm_finetuned",
    "semantic_cache_memmap",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_e_by_d",
    "uses_full_edge_index_on_gpu",
    "uses_dense_nxc_teacher_cache",
    "uses_dense_synthetic_adjacency",
    "uses_fullgraph_edge_backprop",
    "uses_all_pair",
    "student_model",
    "hidden_dim",
    "epochs",
    "soft_temperature",
    "lambda_soft",
    "lambda_hard",
    "lambda_prior",
    "lambda_cover",
    "lambda_calib",
    "lambda_mix",
    "teacher_ensemble_size",
    "teacher_accuracy_each",
    "teacher_valid_acc_each",
    "teacher_entropy_each",
    "teacher_pairwise_kl_mean",
    "teacher_pairwise_kl_min",
    "teacher_pairwise_kl_max",
    "teacher_disagreement_mean",
    "calibrated_ensemble_accuracy",
    "oracle_ensemble_accuracy_if_available",
    "teacher_temperature",
    "valid_ECE",
    "valid_NLL",
    "teacher_ensemble_diversity_failed",
    "selection_time",
    "training_time",
    "precompute_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "planned_condensed_nodes",
    "selection_reservoir_bytes",
    "estimated_edge_scans",
    "estimated_selection_time",
    "estimated_peak_cpu_ram",
    "estimated_peak_gpu_ram",
    "sft_cache_bytes_estimate",
    "semantic_cache_bytes_estimate_if_any",
    "selection_passes",
    "edge_scans",
    "peak_ram_estimate",
    "zero_predicted_classes",
    "per_class_f1_min",
    "per_class_f1_median",
    "per_class_f1_macro",
    "class_coverage_loss",
    "selected_soft_prior_kl",
    "teacher_soft_prior_kl",
    "balanced_track",
    "official_track",
    "base_predictor",
    "base_accuracy",
    "cns_accuracy",
    "graph_direction",
    "normalization_mode",
    "self_loop_mode",
    "logits_cache_hash",
    "feature_checksum",
    "edge_checksum",
    "mask_checksum",
    "teacher_gate_passed",
    "gcrd_accuracy_mean",
    "gcrd_accuracy_std",
    "absolute_pp_gain",
    "relative_accuracy_gain",
    "relative_error_reduction",
    "passes_5pct_error_reduction",
    "passes_absolute_5pp_if_applicable",
    "ratio_definition_match",
    "mathematically_impossible_under_current_teacher_ceiling",
    "teacher_ceiling_gap",
    "notes",
    "next_action",
]


SAFE_FORBIDDEN = (
    "uses_teacher_probs",
    "uses_teacher_logits",
    "uses_logits_as_input",
    "uses_teacher_probs_as_input",
    "uses_external_text_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_e_by_d",
    "uses_full_edge_index_on_gpu",
    "uses_dense_nxc_teacher_cache",
    "uses_dense_synthetic_adjacency",
    "uses_fullgraph_edge_backprop",
    "uses_all_pair",
)
SOTA_FORBIDDEN = (
    "uses_logits_as_input",
    "uses_teacher_probs_as_input",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_e_by_d",
    "uses_full_edge_index_on_gpu",
    "uses_dense_synthetic_adjacency",
    "uses_fullgraph_edge_backprop",
    "uses_all_pair",
)
ULTRA_FORBIDDEN = (
    "uses_dense_nxc_teacher_cache",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_e_by_d",
    "uses_full_edge_index_on_gpu",
    "uses_dense_synthetic_adjacency",
    "uses_fullgraph_edge_backprop",
    "uses_all_pair",
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


def ratio_budget(dataset: str, ratio: float) -> int:
    return max(1, int(round(NUM_NODES[str(dataset)] * float(ratio))))


def default_flags() -> dict[str, bool]:
    return {
        "uses_teacher_probs": False,
        "uses_teacher_logits": False,
        "uses_logits_as_input": False,
        "uses_teacher_probs_as_input": False,
        "soft_target_only": False,
        "uses_external_text_features": False,
        "semantic_features_are_frozen": False,
        "lm_finetuned": False,
        "semantic_cache_memmap": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_e_by_d": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_nxc_teacher_cache": False,
        "uses_dense_synthetic_adjacency": False,
        "uses_fullgraph_edge_backprop": False,
        "uses_all_pair": False,
        "teacher_gate_passed": False,
        "balanced_track": False,
        "official_track": False,
        "teacher_ensemble_diversity_failed": False,
    }


def make_t34_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    condensed_nodes: int | None = None,
    condensed_edges: int = 0,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    status: str = "blocked",
    promotion_track: str = "safe_main",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    notes: str = "",
    next_action: str = "",
    **fields: Any,
) -> dict[str, Any]:
    if condensed_nodes is None:
        condensed_nodes = ratio_budget(dataset, requested_full_node_ratio) if requested_full_node_ratio else 0
    nodes = NUM_NODES.get(str(dataset), 0)
    row: dict[str, Any] = {
        "dataset": dataset,
        "num_nodes": NUM_NODES.get(str(dataset), ""),
        "num_edges": "",
        "num_classes": "",
        "method": method,
        "stage": T34_STAGE,
        "seed": int(seed),
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(condensed_nodes) / float(nodes) if nodes else 0.0,
        "condensed_nodes": int(condensed_nodes),
        "condensed_edges": int(condensed_edges),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "valid_macro_f1": "",
        "valid_test_gap": "",
        "predicted_classes": "",
        "status": status,
        "promotion_track": promotion_track,
        "promotion_status": promotion_status,
        "promotion_allowed": promotion_status == "promoted",
        "failure_reason": failure_reason,
        "teacher_method": "",
        "teacher_accuracy": "",
        "teacher_valid_acc": "",
        "teacher_cache_mode": "",
        "teacher_cache_bytes": "",
        "teacher_dense_cache_bytes_diagnostic": "",
        "teacher_topk_cache_bytes": "",
        "cache_compression_ratio": "",
        "semantic_encoder": "",
        "semantic_cache_path": "",
        "semantic_cache_bytes": "",
        "student_model": "",
        "hidden_dim": "",
        "epochs": "",
        "soft_temperature": "",
        "lambda_soft": "",
        "lambda_hard": "",
        "lambda_prior": "",
        "lambda_cover": "",
        "lambda_calib": "",
        "lambda_mix": "",
        "teacher_ensemble_size": "",
        "teacher_accuracy_each": "",
        "teacher_valid_acc_each": "",
        "teacher_entropy_each": "",
        "teacher_pairwise_kl_mean": "",
        "teacher_pairwise_kl_min": "",
        "teacher_pairwise_kl_max": "",
        "teacher_disagreement_mean": "",
        "calibrated_ensemble_accuracy": "",
        "oracle_ensemble_accuracy_if_available": "",
        "teacher_temperature": "",
        "valid_ECE": "",
        "valid_NLL": "",
        "selection_time": "",
        "training_time": "",
        "precompute_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "planned_condensed_nodes": "",
        "selection_reservoir_bytes": "",
        "estimated_edge_scans": "",
        "estimated_selection_time": "",
        "estimated_peak_cpu_ram": "",
        "estimated_peak_gpu_ram": "",
        "sft_cache_bytes_estimate": "",
        "semantic_cache_bytes_estimate_if_any": "",
        "selection_passes": "",
        "edge_scans": "",
        "peak_ram_estimate": "",
        "zero_predicted_classes": "",
        "per_class_f1_min": "",
        "per_class_f1_median": "",
        "per_class_f1_macro": "",
        "class_coverage_loss": "",
        "selected_soft_prior_kl": "",
        "teacher_soft_prior_kl": "",
        "base_predictor": "",
        "base_accuracy": "",
        "cns_accuracy": "",
        "graph_direction": "",
        "normalization_mode": "",
        "self_loop_mode": "",
        "logits_cache_hash": "",
        "feature_checksum": "",
        "edge_checksum": "",
        "mask_checksum": "",
        "gcrd_accuracy_mean": "",
        "gcrd_accuracy_std": "",
        "absolute_pp_gain": "",
        "relative_accuracy_gain": "",
        "relative_error_reduction": "",
        "passes_5pct_error_reduction": "",
        "passes_absolute_5pp_if_applicable": "",
        "ratio_definition_match": "",
        "mathematically_impossible_under_current_teacher_ceiling": "",
        "teacher_ceiling_gap": "",
        "notes": notes,
        "next_action": next_action,
        **default_flags(),
    }
    row.update(fields)
    for field in T34_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def validate_t34_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = [f"missing_field:{field}" for field in T34_REQUIRED_FIELDS if field not in row]
    track = str(row.get("promotion_track", "safe_main"))
    flags = ULTRA_FORBIDDEN if track == "ultra_planner" else (SOTA_FORBIDDEN if track in {"sota_chase", "semantic_sota"} else SAFE_FORBIDDEN)
    for flag in flags:
        if truthy(row.get(flag, False)):
            forbidden.append(flag)
    if track == "ultra_planner" and str(row.get("teacher_cache_mode", "")) == "dense_fp16":
        forbidden.append("ultra_dense_nxc_teacher_cache")
    if truthy(row.get("uses_dense_nxc_teacher_cache", False)):
        forbidden.append("ultra_dense_nxc_teacher_cache")
    if track in {"sota_chase", "semantic_sota"} and wants_promotion(row):
        if truthy(row.get("uses_teacher_probs", False)) and not truthy(row.get("soft_target_only", False)):
            forbidden.append("soft_target_only_required")
        if str(row.get("teacher_cache_mode", "")) == "" and truthy(row.get("uses_teacher_probs", False)):
            forbidden.append("missing_teacher_cache_mode")
    if wants_promotion(row) and track != "ultra_planner":
        if row.get("accuracy") in {"", None}:
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None}:
            forbidden.append("missing_macro_f1")
        if str(row.get("status", "")) in {"blocked", "failed"} or "carried_forward" in str(row.get("status", "")):
            forbidden.append("status_not_promotable")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t34_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_allowed"] = False
        return guarded
    result = validate_t34_row(guarded)
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
    unsafe = [row for row in promoted if not validate_t34_row(row)["valid"]]
    blocked = Counter(str(row.get("failure_reason", "")) for row in rows if str(row.get("failure_reason", "")))
    return {
        "rows": len(rows),
        "unsafe_promoted_rows": len(unsafe),
        "promoted_safe_rows": sum(1 for row in promoted if row.get("promotion_track") == "safe_main"),
        "promoted_sota_chase_rows": sum(1 for row in promoted if row.get("promotion_track") in {"sota_chase", "semantic_sota"}),
        "promoted_ultra_rows": sum(1 for row in promoted if row.get("promotion_track") == "ultra_planner"),
        "blocked_rows_by_reason": dict(sorted(blocked.items())),
    }


def reddit_stt_gate_status(*, ratio: float, accuracy: float, macro_f1: float) -> tuple[str, str]:
    ratio = float(ratio)
    acc = float(accuracy)
    macro = float(macro_f1)
    if abs(ratio - 0.0005) < 1e-12:
        return ("promoted", "") if acc >= 0.930 else ("not_promoted", "reddit_stt_0p05_accuracy_gate_not_met")
    if abs(ratio - 0.001) < 1e-12:
        return ("promoted", "") if acc >= 0.930 else ("not_promoted", "reddit_stt_0p10_accuracy_gate_not_met")
    if abs(ratio - 0.002) < 1e-12:
        return ("promoted", "") if acc >= 0.936 else ("not_promoted", "reddit_stt_0p20_accuracy_gate_not_met")
    if abs(ratio - 0.0025) < 1e-12:
        return ("promoted", "") if acc >= 0.936 else ("not_promoted", "reddit_stt_0p25_accuracy_gate_not_met")
    if abs(ratio - 0.005) < 1e-12:
        if acc < 0.940:
            return "not_promoted", "reddit_stt_0p50_accuracy_gate_not_met"
        if macro < 0.910:
            return "not_promoted", "reddit_stt_0p50_macro_gate_not_met"
        return "promoted", ""
    if abs(ratio - 0.01) < 1e-12:
        if acc < 0.942:
            return "not_promoted", "reddit_stt_1p00_accuracy_gate_not_met"
        if macro < 0.912:
            return "not_promoted", "reddit_stt_1p00_macro_gate_not_met"
        return "promoted", ""
    return "not_promoted", "reddit_stt_ratio_gate_not_defined"
