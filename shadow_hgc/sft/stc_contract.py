from __future__ import annotations

from typing import Any


T27_STAGE = "t27"

SFT_STC_METHODS: tuple[str, ...] = (
    "sft_stc_frozen_init",
    "sft_stc_trainable_delta",
    "sft_stc_gradient_matching",
    "sft_stc_outer_loop",
    "sft_stc_outer_loop_plus_coverage",
    "sft_stc_gm_plus_coverage",
    "sft_stc_bonsai_sketch_backup",
)

T25_T26_DIAGNOSTIC_METHODS: tuple[str, ...] = (
    "sft_hnr_random",
    "sft_hnr_fdm_herding",
    "sft_hnr_fdm_kcenter",
    "sft_hnr_fdm_hybrid",
    "sft_hnr_fdm_shadow_b1",
    "sft_hnr_fdm_shadow_b2",
)

T27_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_valid_labels",
    "uses_test_labels",
)

T27_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "num_train_nodes",
    "num_classes",
    "init_method",
    "stc_objective",
    "stc_delta_rho",
    "trainable_delta",
    "learn_syn_weights",
    "trainable_labels",
    "syn_rows",
    "syn_feature_dim",
    "syn_weight_mode",
    "inner_steps",
    "outer_steps",
    "gm_num_heads",
    "gm_real_batch_size",
    "head_type",
    "head_hidden_dim",
    "lambda_gm",
    "lambda_mmd",
    "lambda_moment",
    "lambda_div",
    "lambda_prior",
    "lambda_block",
    "lambda_coverage",
    "coverage_track",
    "coverage_prior",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "per_class_f1_min",
    "per_class_f1_median",
    "per_class_f1_max",
    "selected_or_syn_class_count_min",
    "selected_or_syn_class_count_median",
    "selected_or_syn_class_count_max",
    "precompute_time",
    "init_time",
    "stc_optimization_time",
    "final_training_time",
    "total_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_valid_labels",
    "uses_test_labels",
    "uses_all_target_features",
    "promotion_allowed",
    "promotion_status",
    "failure_reason",
    "notes",
    "target_prototypes",
    "shadow_nodes",
    "total_condensed_nodes",
    "condensed_edges",
    "predicted_class_histogram_json",
    "synthetic_class_histogram_json",
    "coverage_gap_before",
    "coverage_gap_after",
    "official_accuracy_track_passed",
    "balanced_robustness_track_passed",
    "uses_year_metadata",
    "enable_temporal_labelreuse_decay",
    "temporal_decay_gamma",
    "valid_acc",
    "A1_passed",
    "A2_passed",
    "A3_passed",
    "teacher_gate_status",
    "status",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _default_flags() -> dict[str, bool]:
    return {
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_valid_labels": False,
        "uses_test_labels": False,
        "uses_all_target_features": False,
    }


def _default_regularizers() -> dict[str, float]:
    return {
        "lambda_gm": 1.0,
        "lambda_mmd": 0.1,
        "lambda_moment": 0.1,
        "lambda_div": 0.01,
        "lambda_prior": 0.1,
        "lambda_block": 0.1,
        "lambda_coverage": 0.0,
    }


def make_t27_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float,
    original_num_nodes: int,
    num_train_nodes: int,
    num_classes: int,
    syn_rows: int,
    syn_feature_dim: int,
    init_method: str = "",
    stc_objective: str = "",
    stc_delta_rho: float | str = "",
    trainable_delta: bool = False,
    learn_syn_weights: bool = False,
    trainable_labels: bool = False,
    syn_weight_mode: str = "none",
    inner_steps: int | str = "",
    outer_steps: int | str = "",
    gm_num_heads: int | str = "",
    gm_real_batch_size: int | str = "",
    head_type: str = "hidden_mlp",
    head_hidden_dim: int | str = "",
    accuracy: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
    per_class_f1_min: float | str = "",
    per_class_f1_median: float | str = "",
    per_class_f1_max: float | str = "",
    status: str = "ready",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    notes: str = "",
    extra: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if int(original_num_nodes) <= 0:
        raise ValueError("original_num_nodes must be positive")
    actual_ratio = int(syn_rows) / float(original_num_nodes)
    row: dict[str, Any] = {
        "dataset": dataset,
        "stage": T27_STAGE,
        "method": method,
        "seed": int(seed),
        "ratio_mode": "full_node",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": actual_ratio,
        "original_num_nodes": int(original_num_nodes),
        "num_train_nodes": int(num_train_nodes),
        "num_classes": int(num_classes),
        "init_method": init_method,
        "stc_objective": stc_objective,
        "stc_delta_rho": stc_delta_rho,
        "trainable_delta": bool(trainable_delta),
        "learn_syn_weights": bool(learn_syn_weights),
        "trainable_labels": bool(trainable_labels),
        "syn_rows": int(syn_rows),
        "syn_feature_dim": int(syn_feature_dim),
        "syn_weight_mode": syn_weight_mode,
        "inner_steps": inner_steps,
        "outer_steps": outer_steps,
        "gm_num_heads": gm_num_heads,
        "gm_real_batch_size": gm_real_batch_size,
        "head_type": head_type,
        "head_hidden_dim": head_hidden_dim,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "per_class_f1_min": per_class_f1_min,
        "per_class_f1_median": per_class_f1_median,
        "per_class_f1_max": per_class_f1_max,
        "selected_or_syn_class_count_min": "",
        "selected_or_syn_class_count_median": "",
        "selected_or_syn_class_count_max": "",
        "precompute_time": "",
        "init_time": "",
        "stc_optimization_time": "",
        "final_training_time": "",
        "total_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "target_prototypes": int(syn_rows),
        "shadow_nodes": 0,
        "total_condensed_nodes": int(syn_rows),
        "condensed_edges": 0,
        "predicted_class_histogram_json": "",
        "synthetic_class_histogram_json": "",
        "coverage_gap_before": "",
        "coverage_gap_after": "",
        "official_accuracy_track_passed": False,
        "balanced_robustness_track_passed": False,
        "uses_year_metadata": False,
        "enable_temporal_labelreuse_decay": False,
        "temporal_decay_gamma": "",
        "valid_acc": "",
        "A1_passed": False,
        "A2_passed": False,
        "A3_passed": False,
        "teacher_gate_status": "",
        "status": status,
        "promotion_allowed": promotion_status == "promoted",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": notes,
        **_default_regularizers(),
        **_default_flags(),
    }
    if extra:
        row.update(extra)
    row.update(overrides)
    for field in T27_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return apply_t27_promotion_guard(row, dataset_gate_passed=promotion_status == "promoted")


def validate_t27_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    for flag in T27_FORBIDDEN_PROMOTED_FLAGS:
        if _truthy(row.get(flag, False)):
            forbidden.append(flag)
    if str(row.get("ratio_mode", "")) != "full_node":
        forbidden.append("ratio_mode_not_full_node")
    if int(float(row.get("shadow_nodes", 0) or 0)) != 0:
        forbidden.append("stc_shadow_nodes_nonzero")
    if int(float(row.get("condensed_edges", 0) or 0)) not in {0, int(float(row.get("syn_rows", 0) or 0))}:
        forbidden.append("stc_condensed_edges_not_zero_or_self_loops")
    if str(row.get("method", "")) in T25_T26_DIAGNOSTIC_METHODS:
        forbidden.append("hnr_fdm_demoted_to_diagnostic")
    if row.get("accuracy", "") in {"", None}:
        forbidden.append("missing_accuracy")
    if row.get("macro_f1", "") in {"", None}:
        forbidden.append("missing_macro_f1")
    if row.get("predicted_classes", "") in {"", None}:
        forbidden.append("missing_predicted_classes")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t27_promotion_guard(row: dict[str, Any], *, dataset_gate_passed: bool) -> dict[str, Any]:
    guarded = dict(row)
    wants_promotion = guarded.get("promotion_status") == "promoted" or _truthy(guarded.get("promotion_allowed", False))
    safety = validate_t27_promoted_row(guarded)
    if not wants_promotion:
        guarded["promotion_allowed"] = False
        guarded.setdefault("failure_reason", "")
        return guarded
    if not safety["valid"]:
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["failure_reason"] = ",".join(safety["forbidden_flags"])
        return guarded
    if not dataset_gate_passed:
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = "not_promoted"
        guarded["failure_reason"] = guarded.get("failure_reason") or "acceptance_gate_not_met"
        return guarded
    guarded["promotion_allowed"] = True
    guarded["promotion_status"] = "promoted"
    guarded["failure_reason"] = guarded.get("failure_reason", "")
    return guarded


def summarize_t27_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if _truthy(row.get("promotion_allowed", False))]
    forbidden_promoted = [row for row in promoted if not validate_t27_promoted_row(row)["valid"]]
    return {
        "rows": len(rows),
        "promoted_rows": len(promoted),
        "forbidden_promoted_rows": len(forbidden_promoted),
        "all_promoted_safe": not forbidden_promoted,
    }
