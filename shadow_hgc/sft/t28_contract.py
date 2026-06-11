from __future__ import annotations

from typing import Any


T28_STAGE = "t28"

ARXIV_NUM_NODES = 169_343
ARXIV_NUM_TRAIN = 90_941
ARXIV_NUM_CLASSES = 40

REDDIT_NUM_NODES = 232_965
REDDIT_NUM_TRAIN = 153_431
REDDIT_NUM_CLASSES = 41

PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_TRAIN = 196_615
PRODUCTS_NUM_CLASSES = 47

ARXIV_A1 = 0.715
ARXIV_A2 = 0.725
ARXIV_A3 = 0.735
ARXIV_SOTA_TEACHER = 0.740

REDDIT_LOW_FIRST_ACC = 0.912
REDDIT_LOW_STRETCH_ACC = 0.916
REDDIT_LOW_T25_REPRO_ACC = 0.921
REDDIT_LOW_T25_REPRO_MACRO = 0.884
REDDIT_MEDIUM_ACC = 0.928
REDDIT_MEDIUM_MACRO = 0.890
REDDIT_ONE_PERCENT_ACC = 0.932
REDDIT_ONE_PERCENT_MACRO = 0.895

ARXIV_TEACHER_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "status",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "valid_acc",
    "teacher_gate_A1_passed",
    "teacher_gate_A2_passed",
    "teacher_gate_A3_passed",
    "teacher_gate_sota_passed",
    "uses_cns_postprocess",
    "uses_temporal_features",
    "uses_temporal_label_decay",
    "uses_fullgraph_gnn_teacher",
    "uses_gnn_hidden_blocks",
    "upper_bound_diagnostic",
    "uses_logits_as_input",
    "uses_teacher_logits_for_condensation",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "cns_correction_alpha",
    "cns_smoothing_alpha",
    "cns_correction_steps",
    "cns_smoothing_steps",
    "cns_autoscale",
    "temporal_decay_gamma",
    "year_feature_dim",
    "precompute_time",
    "training_time",
    "postprocess_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "promotion_allowed",
    "promotion_status",
    "failure_reason",
    "notes",
]

REDDIT_STRUCTURE_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "status",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "num_train_nodes",
    "num_classes",
    "prototype_selector",
    "edge_builder",
    "student_model",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "valid_acc",
    "target_prototypes",
    "shadow_nodes",
    "synthetic_rows",
    "total_condensed_nodes",
    "condensed_edges",
    "edge_topk",
    "edge_symmetry",
    "edge_weight_normalization",
    "uses_knn_graph",
    "uses_cooccur_graph",
    "uses_edge_predictor",
    "uses_ctc_selection",
    "uses_hnr_fdm_control",
    "uses_processed_data_pt",
    "loads_edge_index",
    "uses_lazy_memmap",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_teacher_logits",
    "uses_teacher_logits_for_condensation",
    "uses_kd",
    "uses_dense_p2",
    "uses_logits_as_input",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "precompute_time",
    "condensation_time",
    "edge_build_time",
    "student_training_time",
    "eval_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "full_edge_scans",
    "byte_compression",
    "reddit_raw_edge_stream_used",
    "materialized_stacked_edge_index",
    "edge_candidate_count",
    "edge_predictor_train_pairs",
    "edge_predictor_pos_rate",
    "cooccur_sketch_size",
    "knn_signature_dim",
    "ctc_num_buckets",
    "promotion_allowed",
    "promotion_status",
    "failure_reason",
    "notes",
]

PRODUCTS_MAINTENANCE_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "per_class_f1_json",
    "predicted_hist_json",
    "selected_class_hist_json",
    "official_accuracy_track",
    "balanced_robustness_track",
    "promotion_allowed",
    "promotion_status",
    "failure_reason_if_not_promoted",
    "status",
    "notes",
]

T28_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_teacher_logits_for_condensation",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _has_metric(value: Any) -> bool:
    return value not in {"", None}


def _fvalue(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_safety_flags() -> dict[str, bool]:
    return {
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_teacher_logits_for_condensation": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_processed_data_pt": False,
        "loads_edge_index": False,
        "uses_lazy_memmap": True,
    }


def teacher_gate_flags(accuracy: Any) -> dict[str, bool]:
    acc = _fvalue(accuracy)
    return {
        "teacher_gate_A1_passed": bool(acc >= ARXIV_A1),
        "teacher_gate_A2_passed": bool(acc >= ARXIV_A2),
        "teacher_gate_A3_passed": bool(acc >= ARXIV_A3),
        "teacher_gate_sota_passed": bool(acc >= ARXIV_SOTA_TEACHER),
    }


def make_arxiv_teacher_row(
    *,
    method: str,
    seed: int,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
    valid_acc: float | str = "",
    status: str = "server_ready_not_run",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    notes: str = "",
    uses_cns_postprocess: bool = False,
    uses_temporal_features: bool = False,
    uses_temporal_label_decay: bool = False,
    uses_fullgraph_gnn_teacher: bool = False,
    uses_gnn_hidden_blocks: bool = False,
    upper_bound_diagnostic: bool = False,
    cns_correction_alpha: float | str = "",
    cns_smoothing_alpha: float | str = "",
    cns_correction_steps: int | str = "",
    cns_smoothing_steps: int | str = "",
    cns_autoscale: bool | str = "",
    temporal_decay_gamma: float | str = "",
    year_feature_dim: int | str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": "ogbn-arxiv",
        "stage": T28_STAGE,
        "method": method,
        "seed": int(seed),
        "status": status,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "valid_acc": valid_acc,
        "uses_cns_postprocess": bool(uses_cns_postprocess),
        "uses_temporal_features": bool(uses_temporal_features),
        "uses_temporal_label_decay": bool(uses_temporal_label_decay),
        "uses_fullgraph_gnn_teacher": bool(uses_fullgraph_gnn_teacher),
        "uses_gnn_hidden_blocks": bool(uses_gnn_hidden_blocks),
        "upper_bound_diagnostic": bool(upper_bound_diagnostic),
        "cns_correction_alpha": cns_correction_alpha,
        "cns_smoothing_alpha": cns_smoothing_alpha,
        "cns_correction_steps": cns_correction_steps,
        "cns_smoothing_steps": cns_smoothing_steps,
        "cns_autoscale": cns_autoscale,
        "temporal_decay_gamma": temporal_decay_gamma,
        "year_feature_dim": year_feature_dim,
        "precompute_time": "",
        "training_time": "",
        "postprocess_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "promotion_allowed": promotion_status == "promoted",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": notes,
        **_default_safety_flags(),
        **teacher_gate_flags(accuracy),
    }
    if extra:
        row.update(extra)
    for field in ARXIV_TEACHER_FIELDS:
        row.setdefault(field, "")
    return row


def make_reddit_structure_row(
    *,
    method: str,
    seed: int,
    requested_full_node_ratio: float,
    original_num_nodes: int = REDDIT_NUM_NODES,
    num_train_nodes: int = REDDIT_NUM_TRAIN,
    num_classes: int = REDDIT_NUM_CLASSES,
    target_prototypes: int = 0,
    shadow_nodes: int = 0,
    synthetic_rows: int = 0,
    condensed_edges: int = 0,
    prototype_selector: str = "",
    edge_builder: str = "",
    student_model: str = "",
    edge_topk: int | str = "",
    edge_symmetry: str = "",
    edge_weight_normalization: str = "dst_row",
    accuracy: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
    valid_acc: float | str = "",
    status: str = "server_ready_not_run",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    notes: str = "",
    extra: dict[str, Any] | None = None,
    **flags: Any,
) -> dict[str, Any]:
    if int(original_num_nodes) <= 0:
        raise ValueError("original_num_nodes must be positive")
    total_nodes = int(target_prototypes) + int(shadow_nodes) + int(synthetic_rows)
    actual = total_nodes / float(original_num_nodes)
    row: dict[str, Any] = {
        "dataset": "Reddit",
        "stage": T28_STAGE,
        "method": method,
        "seed": int(seed),
        "status": status,
        "ratio_mode": "full_node",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": actual,
        "original_num_nodes": int(original_num_nodes),
        "num_train_nodes": int(num_train_nodes),
        "num_classes": int(num_classes),
        "prototype_selector": prototype_selector,
        "edge_builder": edge_builder,
        "student_model": student_model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "valid_acc": valid_acc,
        "target_prototypes": int(target_prototypes),
        "shadow_nodes": int(shadow_nodes),
        "synthetic_rows": int(synthetic_rows),
        "total_condensed_nodes": total_nodes,
        "condensed_edges": int(condensed_edges),
        "edge_topk": edge_topk,
        "edge_symmetry": edge_symmetry,
        "edge_weight_normalization": edge_weight_normalization,
        "uses_knn_graph": edge_builder == "knn",
        "uses_cooccur_graph": edge_builder == "cooccur",
        "uses_edge_predictor": edge_builder == "edge_predictor",
        "uses_ctc_selection": "ctc" in prototype_selector,
        "uses_hnr_fdm_control": "hnr_fdm" in prototype_selector or "hnr_fdm" in method,
        "precompute_time": "",
        "condensation_time": "",
        "edge_build_time": "",
        "student_training_time": "",
        "eval_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "full_edge_scans": 0,
        "byte_compression": "",
        "reddit_raw_edge_stream_used": True,
        "materialized_stacked_edge_index": False,
        "edge_candidate_count": "",
        "edge_predictor_train_pairs": "",
        "edge_predictor_pos_rate": "",
        "cooccur_sketch_size": "",
        "knn_signature_dim": "",
        "ctc_num_buckets": "",
        "promotion_allowed": promotion_status == "promoted",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": notes,
        **_default_safety_flags(),
    }
    row.update(flags)
    if extra:
        row.update(extra)
    for field in REDDIT_STRUCTURE_FIELDS:
        row.setdefault(field, "")
    return row


def make_products_maintenance_row(
    *,
    method: str,
    seed: int,
    requested_full_node_ratio: float,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
    status: str = "carried_forward",
    promotion_status: str = "carry_forward",
    failure_reason_if_not_promoted: str = "",
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = max(1, int(round(float(requested_full_node_ratio) * PRODUCTS_NUM_NODES)))
    row: dict[str, Any] = {
        "dataset": "ogbn-products",
        "stage": T28_STAGE,
        "method": method,
        "seed": int(seed),
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": total / float(PRODUCTS_NUM_NODES),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "per_class_f1_json": "",
        "predicted_hist_json": "",
        "selected_class_hist_json": "",
        "official_accuracy_track": "",
        "balanced_robustness_track": "",
        "promotion_allowed": False,
        "promotion_status": promotion_status,
        "failure_reason_if_not_promoted": failure_reason_if_not_promoted,
        "status": status,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    for field in PRODUCTS_MAINTENANCE_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t28_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    for flag in T28_FORBIDDEN_PROMOTED_FLAGS:
        if _truthy(row.get(flag, False)):
            forbidden.append(flag)
    if str(row.get("dataset")) == "Reddit" and _truthy(row.get("uses_processed_data_pt", False)):
        forbidden.append("uses_processed_data_pt")
    if str(row.get("dataset")) == "Reddit" and str(row.get("ratio_mode", "")) != "full_node":
        forbidden.append("ratio_mode_not_full_node")
    if str(row.get("dataset")) == "ogbn-arxiv":
        if _truthy(row.get("upper_bound_diagnostic", False)):
            forbidden.append("upper_bound_diagnostic_not_scalable_main")
        if _truthy(row.get("uses_fullgraph_gnn_teacher", False)) and not _truthy(row.get("upper_bound_diagnostic", False)):
            forbidden.append("fullgraph_gnn_teacher_not_marked_upper_bound")
    if not _has_metric(row.get("accuracy")):
        forbidden.append("missing_accuracy")
    if not _has_metric(row.get("macro_f1")):
        forbidden.append("missing_macro_f1")
    if not _has_metric(row.get("predicted_classes")):
        forbidden.append("missing_predicted_classes")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def _arxiv_gate_passed(row: dict[str, Any]) -> bool:
    return _truthy(row.get("teacher_gate_A1_passed", False)) or _fvalue(row.get("accuracy")) >= ARXIV_A1


def apply_t28_promotion_guard(row: dict[str, Any], *, dataset_gate_passed: bool) -> dict[str, Any]:
    guarded = dict(row)
    wants_promotion = guarded.get("promotion_status") == "promoted" or _truthy(guarded.get("promotion_allowed", False))
    if not wants_promotion:
        guarded["promotion_allowed"] = False
        return guarded
    safety = validate_t28_promoted_row(guarded)
    if not safety["valid"]:
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["failure_reason"] = ",".join(safety["forbidden_flags"])
        return guarded
    if str(guarded.get("dataset")) == "ogbn-arxiv" and not _arxiv_gate_passed(guarded):
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = "blocked_teacher_gate"
        guarded["failure_reason"] = "arxiv_teacher_below_A1"
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


def reddit_gate_passed(row: dict[str, Any]) -> bool:
    ratio = _fvalue(row.get("requested_full_node_ratio"))
    acc = _fvalue(row.get("accuracy"))
    macro = _fvalue(row.get("macro_f1"))
    if abs(ratio - 0.001) < 1e-12:
        return acc >= REDDIT_LOW_STRETCH_ACC
    if abs(ratio - 0.005) < 1e-12:
        return acc >= REDDIT_MEDIUM_ACC and macro >= REDDIT_MEDIUM_MACRO
    if abs(ratio - 0.01) < 1e-12:
        return acc >= REDDIT_ONE_PERCENT_ACC and macro >= REDDIT_ONE_PERCENT_MACRO
    return False


def summarize_t28_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if _truthy(row.get("promotion_allowed", False)) or row.get("promotion_status") == "promoted"]
    forbidden = [row for row in promoted if not validate_t28_promoted_row(row)["valid"]]
    return {
        "rows": len(rows),
        "promoted_rows": len(promoted),
        "forbidden_promoted_rows": len(forbidden),
        "all_promoted_safe": not forbidden,
    }
