from __future__ import annotations

from typing import Any


T29_STAGE = "t29"

ARXIV_NUM_NODES = 169_343
ARXIV_NUM_CLASSES = 40
REDDIT_NUM_NODES = 232_965
REDDIT_NUM_CLASSES = 41
PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_CLASSES = 47

ARXIV_A1 = 0.715
ARXIV_A2 = 0.725
ARXIV_A3 = 0.740

REDDIT_OMCP_001_ACC = 0.923
REDDIT_OMCP_001_MACRO = 0.886
REDDIT_OMCP_005_ACC = 0.928
REDDIT_OMCP_005_MACRO = 0.890

T29_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "status",
    "promotion_status",
    "promotion_track",
    "failure_reason",
    "notes",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "actual_condensed_nodes",
    "target_prototypes",
    "shadow_nodes",
    "extra_synthetic_nodes",
    "total_condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "teacher_accuracy",
    "teacher_valid_acc",
    "teacher_macro_f1",
    "teacher_method",
    "precompute_time",
    "condensation_time",
    "operator_fit_time",
    "training_time",
    "eval_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "full_edge_scans",
    "uses_processed_data_pt",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_exact_pairwise",
    "uses_all_target_cache",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_valid_labels_for_calibration",
    "uses_external_text_features",
    "uses_raw_text",
    "uses_lm_encoder",
    "uses_cns_postprocess",
    "cns_correction_alpha",
    "cns_smoothing_alpha",
    "cns_correction_steps",
    "cns_smoothing_steps",
    "cns_autoscale",
    "operator_topk",
    "operator_candidate_edges",
    "operator_edges",
    "operator_loss_x1",
    "operator_loss_x2",
    "operator_loss_y1",
    "operator_loss_y2",
    "operator_row_sum_error",
    "operator_negative_weight_count",
    "operator_entropy",
    "student_model",
    "pltc_num_soft_nodes",
    "pltc_num_hard_train_nodes",
    "pltc_temperature",
    "pltc_confidence_min",
    "pltc_confidence_max",
    "pltc_confidence_bins",
    "pltc_soft_class_coverage",
    "semantic_lm_model",
    "semantic_feature_dim",
    "semantic_cache_bytes",
    "semantic_encode_time",
    "source_table",
]

SAFE_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_exact_pairwise",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)

SOTA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fvalue(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def original_num_nodes(dataset: str) -> int:
    name = str(dataset)
    if name == "Reddit":
        return REDDIT_NUM_NODES
    if name == "ogbn-arxiv":
        return ARXIV_NUM_NODES
    if name == "ogbn-products":
        return PRODUCTS_NUM_NODES
    raise ValueError(f"unknown T29 dataset: {dataset}")


def num_classes(dataset: str) -> int:
    name = str(dataset)
    if name == "Reddit":
        return REDDIT_NUM_CLASSES
    if name == "ogbn-arxiv":
        return ARXIV_NUM_CLASSES
    if name == "ogbn-products":
        return PRODUCTS_NUM_CLASSES
    raise ValueError(f"unknown T29 dataset: {dataset}")


def ratio_budget(dataset: str, requested_full_node_ratio: float) -> int:
    nodes = original_num_nodes(dataset)
    return max(1, int(round(float(requested_full_node_ratio) * nodes)))


def default_flags() -> dict[str, bool]:
    return {
        "uses_processed_data_pt": False,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_adjacency": False,
        "uses_exact_pairwise": False,
        "uses_all_target_cache": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_valid_labels_for_calibration": False,
        "uses_external_text_features": False,
        "uses_raw_text": False,
        "uses_lm_encoder": False,
        "uses_cns_postprocess": False,
    }


def make_t29_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    original_num_nodes: int | None = None,
    target_prototypes: int = 0,
    shadow_nodes: int = 0,
    extra_synthetic_nodes: int = 0,
    total_condensed_edges: int = 0,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    predicted_classes: int | str = "",
    teacher_accuracy: float | str = "",
    teacher_valid_acc: float | str = "",
    teacher_macro_f1: float | str = "",
    teacher_method: str = "",
    status: str = "server_ready_not_run",
    promotion_status: str = "not_promoted",
    promotion_track: str = "safe_mainline",
    failure_reason: str = "",
    notes: str = "",
    source_table: str = "",
    extra: dict[str, Any] | None = None,
    **flags: Any,
) -> dict[str, Any]:
    nodes = int(original_num_nodes if original_num_nodes is not None else globals()["original_num_nodes"](dataset))
    actual_nodes = int(target_prototypes) + int(shadow_nodes) + int(extra_synthetic_nodes)
    row: dict[str, Any] = {
        "dataset": dataset,
        "stage": T29_STAGE,
        "method": method,
        "seed": int(seed),
        "status": status,
        "promotion_status": promotion_status,
        "promotion_track": promotion_track,
        "failure_reason": failure_reason,
        "notes": notes,
        "ratio_mode": "full_node",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(actual_nodes) / float(nodes) if nodes > 0 else 0.0,
        "original_num_nodes": nodes,
        "actual_condensed_nodes": int(actual_nodes),
        "target_prototypes": int(target_prototypes),
        "shadow_nodes": int(shadow_nodes),
        "extra_synthetic_nodes": int(extra_synthetic_nodes),
        "total_condensed_edges": int(total_condensed_edges),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "teacher_accuracy": teacher_accuracy,
        "teacher_valid_acc": teacher_valid_acc,
        "teacher_macro_f1": teacher_macro_f1,
        "teacher_method": teacher_method,
        "precompute_time": "",
        "condensation_time": "",
        "operator_fit_time": "",
        "training_time": "",
        "eval_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "full_edge_scans": 0,
        "operator_topk": "",
        "cns_correction_alpha": "",
        "cns_smoothing_alpha": "",
        "cns_correction_steps": "",
        "cns_smoothing_steps": "",
        "cns_autoscale": "",
        "operator_candidate_edges": "",
        "operator_edges": "",
        "operator_loss_x1": "",
        "operator_loss_x2": "",
        "operator_loss_y1": "",
        "operator_loss_y2": "",
        "operator_row_sum_error": "",
        "operator_negative_weight_count": "",
        "operator_entropy": "",
        "student_model": "",
        "pltc_num_soft_nodes": "",
        "pltc_num_hard_train_nodes": "",
        "pltc_temperature": "",
        "pltc_confidence_min": "",
        "pltc_confidence_max": "",
        "pltc_confidence_bins": "",
        "pltc_soft_class_coverage": "",
        "semantic_lm_model": "",
        "semantic_feature_dim": "",
        "semantic_cache_bytes": "",
        "semantic_encode_time": "",
        "source_table": source_table,
        **default_flags(),
    }
    row.update(flags)
    if extra:
        row.update(extra)
    for field in T29_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def ratio_within_tolerance(row: dict[str, Any]) -> bool:
    requested = fvalue(row.get("requested_full_node_ratio"))
    if requested == 0.0:
        return True
    actual = fvalue(row.get("actual_full_node_ratio"))
    original = max(1, int(fvalue(row.get("original_num_nodes"), 1.0)))
    tolerance = max(1.0 / float(original), 0.05 * requested)
    return abs(actual - requested) <= tolerance


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def validate_t29_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    missing = [field for field in T29_REQUIRED_FIELDS if field not in row]
    forbidden.extend(f"missing_field:{field}" for field in missing)
    track = str(row.get("promotion_track", "safe_mainline"))
    flags = SOTA_FORBIDDEN_FLAGS if track == "sota_chase" else SAFE_FORBIDDEN_FLAGS
    for flag in flags:
        if truthy(row.get(flag, False)):
            forbidden.append(flag)
    if track == "safe_mainline" and str(row.get("dataset")) == "Reddit" and truthy(row.get("uses_processed_data_pt", False)):
        forbidden.append("uses_processed_data_pt")
    if not ratio_within_tolerance(row):
        forbidden.append("ratio_mismatch")
    if wants_promotion(row):
        if row.get("accuracy") in {"", None}:
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None} and str(row.get("dataset")) != "ogbn-products":
            forbidden.append("missing_macro_f1")
        if row.get("predicted_classes") in {"", None}:
            forbidden.append("missing_predicted_classes")
        status = str(row.get("status", ""))
        if not (status.startswith("completed_long") or status.startswith("completed_real")):
            forbidden.append("status_not_completed_real_or_long")
        if str(row.get("method", "")).startswith("reddit_sft_omcp"):
            if int(fvalue(row.get("operator_negative_weight_count"), 1.0)) != 0:
                forbidden.append("operator_has_negative_weights")
            if fvalue(row.get("operator_row_sum_error"), 1.0) >= 1e-4:
                forbidden.append("operator_row_sum_error")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t29_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_status"] = guarded.get("promotion_status") or "not_promoted"
        return guarded
    result = validate_t29_row(guarded)
    if not result["valid"]:
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["promotion_allowed"] = False
        guarded["failure_reason"] = ",".join(result["forbidden_flags"])
        return guarded
    guarded["promotion_status"] = "promoted"
    guarded["promotion_allowed"] = True
    return guarded


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if wants_promotion(row)]
    unsafe = [row for row in promoted if not validate_t29_row(row)["valid"]]
    return {
        "rows": len(rows),
        "promoted_rows": len(promoted),
        "unsafe_promoted_rows": len(unsafe),
        "all_promoted_safe": not unsafe,
    }
