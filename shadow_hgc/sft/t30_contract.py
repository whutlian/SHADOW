from __future__ import annotations

from collections import Counter
from typing import Any


T30_STAGE = "t30"

ARXIV_NUM_NODES = 169_343
ARXIV_NUM_CLASSES = 40
REDDIT_NUM_NODES = 232_965
REDDIT_NUM_CLASSES = 41
PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_CLASSES = 47

ARXIV_A1 = 0.715
ARXIV_A2 = 0.725
ARXIV_A3 = 0.740

REDDIT_SAFE_001 = 0.923
REDDIT_SAFE_005_ACC = 0.928
REDDIT_SAFE_005_MACRO = 0.890
REDDIT_SOTA_001 = 0.926
REDDIT_SOTA_005 = 0.932

T30_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "promotion_track",
    "promotion_status",
    "promotion_allowed",
    "status",
    "failure_reason",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "total_condensed_nodes",
    "total_condensed_edges",
    "num_codewords",
    "num_labeled_codewords",
    "num_unlabeled_codewords",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "precompute_time",
    "assignment_time",
    "operator_build_time",
    "condensation_time",
    "training_time",
    "eval_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "full_edge_scans",
    "assignment_mode",
    "operator_mode",
    "operator_topk",
    "operator_edges_before_topk",
    "operator_edges_after_topk",
    "operator_row_sum_error",
    "operator_zero_rows",
    "operator_repaired_rows",
    "operator_entropy",
    "operator_max_weight",
    "operator_min_nonzero_weight",
    "quotient_build_mode",
    "student_model",
    "transfer_eval_type",
    "uses_teacher_logits",
    "uses_kd",
    "uses_external_text_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_dense_adjacency",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_exact_pairwise",
    "uses_processed_data_pt",
    "uses_raw_text",
    "uses_lm_encoder",
    "uses_cns_postprocess",
    "base_predictor",
    "base_valid_acc",
    "base_test_acc",
    "cns_valid_acc",
    "cns_test_acc",
    "cns_correction_alpha",
    "cns_smoothing_alpha",
    "cns_correction_steps",
    "cns_smoothing_steps",
    "teacher_name",
    "teacher_accuracy_if_known",
    "teacher_calibration_temperature",
    "soft_label_entropy_mean",
    "confidence_bin_counts",
    "semantic_lm_model",
    "semantic_feature_dim",
    "semantic_cache_bytes",
    "semantic_manifest",
    "class_histogram_json",
    "notes",
    "next_action",
    "source_table",
]

SAFE_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_dense_adjacency",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_exact_pairwise",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_processed_data_pt",
)

SOTA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_dense_adjacency",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_exact_pairwise",
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
    raise ValueError(f"unknown T30 dataset: {dataset}")


def num_classes(dataset: str) -> int:
    name = str(dataset)
    if name == "Reddit":
        return REDDIT_NUM_CLASSES
    if name == "ogbn-arxiv":
        return ARXIV_NUM_CLASSES
    if name == "ogbn-products":
        return PRODUCTS_NUM_CLASSES
    raise ValueError(f"unknown T30 dataset: {dataset}")


def ratio_budget(dataset: str, requested_full_node_ratio: float) -> int:
    return max(1, int(round(original_num_nodes(dataset) * float(requested_full_node_ratio))))


def default_flags() -> dict[str, bool]:
    return {
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_external_text_features": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_dense_p2": False,
        "uses_dense_adjacency": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_exact_pairwise": False,
        "uses_processed_data_pt": False,
        "uses_raw_text": False,
        "uses_lm_encoder": False,
        "uses_cns_postprocess": False,
    }


def make_t30_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    original_nodes: int | None = None,
    num_codewords: int = 0,
    num_labeled_codewords: int = 0,
    num_unlabeled_codewords: int = 0,
    total_condensed_edges: int = 0,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    valid_acc: float | str = "",
    predicted_classes: int | str = "",
    status: str = "blocked",
    promotion_status: str = "not_promoted",
    promotion_track: str = "safe_main",
    failure_reason: str = "",
    notes: str = "",
    transfer_eval_type: str = "",
    student_model: str = "",
    assignment_mode: str = "",
    operator_mode: str = "",
    quotient_build_mode: str = "",
    next_action: str = "",
    source_table: str = "",
    extra: dict[str, Any] | None = None,
    **flags: Any,
) -> dict[str, Any]:
    nodes = int(original_nodes if original_nodes is not None else original_num_nodes(dataset))
    condensed_nodes = int(num_codewords)
    row: dict[str, Any] = {
        "dataset": dataset,
        "stage": T30_STAGE,
        "method": method,
        "seed": int(seed),
        "promotion_track": promotion_track,
        "promotion_status": promotion_status,
        "promotion_allowed": promotion_status == "promoted",
        "status": status,
        "failure_reason": failure_reason,
        "ratio_mode": "full_node",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(condensed_nodes) / float(nodes) if nodes > 0 else 0.0,
        "original_num_nodes": nodes,
        "total_condensed_nodes": condensed_nodes,
        "total_condensed_edges": int(total_condensed_edges),
        "num_codewords": condensed_nodes,
        "num_labeled_codewords": int(num_labeled_codewords),
        "num_unlabeled_codewords": int(num_unlabeled_codewords),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "precompute_time": "",
        "assignment_time": "",
        "operator_build_time": "",
        "condensation_time": "",
        "training_time": "",
        "eval_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "full_edge_scans": 0,
        "assignment_mode": assignment_mode,
        "operator_mode": operator_mode,
        "operator_topk": "",
        "operator_edges_before_topk": "",
        "operator_edges_after_topk": "",
        "operator_row_sum_error": "",
        "operator_zero_rows": "",
        "operator_repaired_rows": "",
        "operator_entropy": "",
        "operator_max_weight": "",
        "operator_min_nonzero_weight": "",
        "quotient_build_mode": quotient_build_mode,
        "student_model": student_model,
        "transfer_eval_type": transfer_eval_type,
        "base_predictor": "",
        "base_valid_acc": "",
        "base_test_acc": "",
        "cns_valid_acc": "",
        "cns_test_acc": "",
        "cns_correction_alpha": "",
        "cns_smoothing_alpha": "",
        "cns_correction_steps": "",
        "cns_smoothing_steps": "",
        "teacher_name": "",
        "teacher_accuracy_if_known": "",
        "teacher_calibration_temperature": "",
        "soft_label_entropy_mean": "",
        "confidence_bin_counts": "",
        "semantic_lm_model": "",
        "semantic_feature_dim": "",
        "semantic_cache_bytes": "",
        "semantic_manifest": "",
        "class_histogram_json": "",
        "notes": notes,
        "next_action": next_action,
        "source_table": source_table,
        **default_flags(),
    }
    row.update(flags)
    if extra:
        row.update(extra)
    for field in T30_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def ratio_within_one_node(row: dict[str, Any]) -> bool:
    requested = fvalue(row.get("requested_full_node_ratio"))
    if requested == 0.0:
        return True
    actual = fvalue(row.get("actual_full_node_ratio"))
    original = max(1, int(fvalue(row.get("original_num_nodes"), 1.0)))
    return abs(actual - requested) <= (1.0 / float(original) + 1e-12)


def is_qoc_row(row: dict[str, Any]) -> bool:
    method = str(row.get("method", "")).lower()
    return "qoc" in method or int(fvalue(row.get("num_codewords"), 0)) > 0


def validate_t30_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    missing = [field for field in T30_REQUIRED_FIELDS if field not in row]
    forbidden.extend(f"missing_field:{field}" for field in missing)
    track = str(row.get("promotion_track", "safe_main"))
    flags = SOTA_FORBIDDEN_FLAGS if track == "sota_chase" else SAFE_FORBIDDEN_FLAGS
    if wants_promotion(row):
        for flag in flags:
            if truthy(row.get(flag, False)):
                forbidden.append(flag)
    if wants_promotion(row):
        if row.get("accuracy") in {"", None}:
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None}:
            forbidden.append("missing_macro_f1")
        if row.get("predicted_classes") in {"", None}:
            forbidden.append("missing_predicted_classes")
        if str(row.get("failure_reason", "")):
            forbidden.append("nonempty_failure_reason")
        status = str(row.get("status", ""))
        if status in {"blocked", "failed"} or "smoke" in status:
            forbidden.append("status_not_promotable")
        if is_qoc_row(row):
            if row.get("transfer_eval_type") != "real_transfer_eval":
                forbidden.append("qoc_requires_real_transfer_eval")
            if fvalue(row.get("operator_row_sum_error"), 1.0) > 1e-4:
                forbidden.append("operator_row_sum_error")
            if not ratio_within_one_node(row):
                forbidden.append("ratio_mismatch")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t30_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_status"] = guarded.get("promotion_status") or "not_promoted"
        guarded["promotion_allowed"] = False
        return guarded
    result = validate_t30_row(guarded)
    if not result["valid"]:
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["promotion_allowed"] = False
        guarded["failure_reason"] = ",".join(result["forbidden_flags"])
        return guarded
    guarded["promotion_status"] = "promoted"
    guarded["promotion_allowed"] = True
    return guarded


def summarize_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if wants_promotion(row)]
    unsafe = [row for row in promoted if not validate_t30_row(row)["valid"]]
    blocked_reasons = Counter(str(row.get("failure_reason", "")) for row in rows if str(row.get("failure_reason", "")))
    return {
        "rows": len(rows),
        "promoted_safe_rows": sum(1 for row in promoted if row.get("promotion_track") == "safe_main"),
        "promoted_sota_chase_rows": sum(1 for row in promoted if row.get("promotion_track") == "sota_chase"),
        "unsafe_promoted_rows": len(unsafe),
        "blocked_rows_by_reason": dict(sorted(blocked_reasons.items())),
    }
