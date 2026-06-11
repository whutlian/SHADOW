from __future__ import annotations

from collections import Counter
from typing import Any


T31_STAGE = "t31"

ARXIV_NUM_NODES = 169_343
ARXIV_NUM_CLASSES = 40
REDDIT_NUM_NODES = 232_965
REDDIT_NUM_CLASSES = 41
PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_CLASSES = 47

REDDIT_TTC_001_GATE = 0.925
REDDIT_TTC_005_GATE = 0.930
REDDIT_SIMSFT_001_GATE = 0.923
REDDIT_SIMSFT_005_GATE = 0.928
ARXIV_RAW_MLP_CNS_SANITY = 0.720
ARXIV_SFT_CNS_GATE = 0.715
ARXIV_SEMANTIC_GATE = 0.740

T31_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "status",
    "failure_reason",
    "promotion_track",
    "promotion_status",
    "promotion_allowed",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "total_condensed_nodes",
    "syn_rows",
    "shadow_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "teacher_method",
    "teacher_accuracy",
    "teacher_macro_f1",
    "teacher_valid_acc",
    "teacher_temperature",
    "teacher_entropy_mean",
    "teacher_margin_mean",
    "teacher_disagreement_mean",
    "teacher_cache_bytes",
    "teacher_logits_cache_path",
    "uses_teacher_logits",
    "soft_label_source",
    "candidate_nodes",
    "candidate_bucket_counts_json",
    "selected_bucket_counts_json",
    "soft_class_mass_coverage",
    "entropy_bucket_coverage",
    "margin_bucket_coverage",
    "degree_bucket_coverage",
    "hard_anchor_count",
    "soft_only_count",
    "mixup_row_count",
    "target_prior_type",
    "student_model",
    "hidden_dim",
    "epochs",
    "dropout",
    "weight_decay",
    "label_smoothing",
    "base_predictor",
    "base_accuracy",
    "base_macro_f1",
    "base_valid_acc",
    "cns_accuracy",
    "cns_macro_f1",
    "cns_valid_acc",
    "uses_cns_postprocess",
    "best_correction_alpha",
    "best_smoothing_alpha",
    "best_correction_steps",
    "best_smoothing_steps",
    "autoscale",
    "graph_direction",
    "uses_external_text_features",
    "semantic_model_name",
    "semantic_feature_dim",
    "semantic_cache_path",
    "semantic_cache_bytes",
    "raw_text_map_path",
    "node_id_to_paper_id_path",
    "semantic_match_rate",
    "semantic_unmatched_nodes",
    "uses_valid_labels_for_hyperparam_selection",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_exact_pairwise",
    "precompute_time",
    "condensation_time",
    "training_time",
    "inference_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "full_edge_scans",
    "per_class_f1_json",
    "selected_class_hist_json",
    "predicted_class_hist_json",
    "byte_compression",
    "source_table",
    "notes",
    "next_action",
]

SAFE_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_teacher_logits",
    "uses_kd",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_exact_pairwise",
)

SOTA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_exact_pairwise",
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
    name = str(dataset)
    if name == "Reddit":
        return REDDIT_NUM_NODES
    if name == "ogbn-arxiv":
        return ARXIV_NUM_NODES
    if name == "ogbn-products":
        return PRODUCTS_NUM_NODES
    raise ValueError(f"unknown T31 dataset: {dataset}")


def num_classes(dataset: str) -> int:
    name = str(dataset)
    if name == "Reddit":
        return REDDIT_NUM_CLASSES
    if name == "ogbn-arxiv":
        return ARXIV_NUM_CLASSES
    if name == "ogbn-products":
        return PRODUCTS_NUM_CLASSES
    raise ValueError(f"unknown T31 dataset: {dataset}")


def ratio_budget(dataset: str, requested_full_node_ratio: float) -> int:
    return max(1, int(round(original_num_nodes(dataset) * float(requested_full_node_ratio))))


def default_flags() -> dict[str, bool]:
    return {
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_external_text_features": False,
        "uses_valid_labels_for_hyperparam_selection": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_adjacency": False,
        "uses_exact_pairwise": False,
        "uses_cns_postprocess": False,
    }


def make_t31_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    original_nodes: int | None = None,
    total_condensed_nodes: int | None = None,
    syn_rows: int | None = None,
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
    student_model: str = "",
    hidden_dim: int | str = "",
    epochs: int | str = "",
    notes: str = "",
    next_action: str = "",
    source_table: str = "",
    extra: dict[str, Any] | None = None,
    **flags: Any,
) -> dict[str, Any]:
    nodes = int(original_nodes if original_nodes is not None else original_num_nodes(dataset))
    if total_condensed_nodes is None:
        total_condensed_nodes = ratio_budget(dataset, requested_full_node_ratio) if requested_full_node_ratio else 0
    if syn_rows is None:
        syn_rows = int(total_condensed_nodes)
    row: dict[str, Any] = {
        "dataset": dataset,
        "stage": T31_STAGE,
        "method": method,
        "seed": int(seed),
        "status": status,
        "failure_reason": failure_reason,
        "promotion_track": promotion_track,
        "promotion_status": promotion_status,
        "promotion_allowed": promotion_status == "promoted",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(total_condensed_nodes) / float(nodes) if nodes else 0.0,
        "original_num_nodes": nodes,
        "total_condensed_nodes": int(total_condensed_nodes),
        "syn_rows": int(syn_rows),
        "shadow_nodes": int(shadow_nodes),
        "condensed_edges": int(condensed_edges),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "teacher_method": "",
        "teacher_accuracy": "",
        "teacher_macro_f1": "",
        "teacher_valid_acc": "",
        "teacher_temperature": "",
        "teacher_entropy_mean": "",
        "teacher_margin_mean": "",
        "teacher_disagreement_mean": "",
        "teacher_cache_bytes": "",
        "teacher_logits_cache_path": "",
        "soft_label_source": "",
        "candidate_nodes": "",
        "candidate_bucket_counts_json": "",
        "selected_bucket_counts_json": "",
        "soft_class_mass_coverage": "",
        "entropy_bucket_coverage": "",
        "margin_bucket_coverage": "",
        "degree_bucket_coverage": "",
        "hard_anchor_count": "",
        "soft_only_count": "",
        "mixup_row_count": "",
        "target_prior_type": "",
        "student_model": student_model,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "dropout": "",
        "weight_decay": "",
        "label_smoothing": "",
        "base_predictor": "",
        "base_accuracy": "",
        "base_macro_f1": "",
        "base_valid_acc": "",
        "cns_accuracy": "",
        "cns_macro_f1": "",
        "cns_valid_acc": "",
        "best_correction_alpha": "",
        "best_smoothing_alpha": "",
        "best_correction_steps": "",
        "best_smoothing_steps": "",
        "autoscale": "",
        "graph_direction": "",
        "semantic_model_name": "",
        "semantic_feature_dim": "",
        "semantic_cache_path": "",
        "semantic_cache_bytes": "",
        "raw_text_map_path": "",
        "node_id_to_paper_id_path": "",
        "semantic_match_rate": "",
        "semantic_unmatched_nodes": "",
        "precompute_time": "",
        "condensation_time": "",
        "training_time": "",
        "inference_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "full_edge_scans": 0,
        "per_class_f1_json": "",
        "selected_class_hist_json": "",
        "predicted_class_hist_json": "",
        "byte_compression": "",
        "source_table": source_table,
        "notes": notes,
        "next_action": next_action,
        **default_flags(),
    }
    row.update(flags)
    if extra:
        row.update(extra)
    for field in T31_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def validate_t31_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    missing = [field for field in T31_REQUIRED_FIELDS if field not in row]
    forbidden.extend(f"missing_field:{field}" for field in missing)
    track = str(row.get("promotion_track", "safe_main"))
    flags = SOTA_FORBIDDEN_FLAGS if track == "sota_chase" else SAFE_FORBIDDEN_FLAGS
    for flag in flags:
        if truthy(row.get(flag, False)):
            forbidden.append(flag)
    if wants_promotion(row):
        if row.get("accuracy") in {"", None}:
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None}:
            forbidden.append("missing_macro_f1")
        status = str(row.get("status", ""))
        if status in {"blocked", "failed"} or "smoke" in status:
            forbidden.append("status_not_promotable")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t31_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_status"] = guarded.get("promotion_status") or "not_promoted"
        guarded["promotion_allowed"] = False
        return guarded
    result = validate_t31_row(guarded)
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
    unsafe = [row for row in promoted if not validate_t31_row(row)["valid"]]
    blocked_reasons = Counter(str(row.get("failure_reason", "")) for row in rows if str(row.get("failure_reason", "")))
    return {
        "rows": len(rows),
        "promoted_safe_rows": sum(1 for row in promoted if row.get("promotion_track") == "safe_main"),
        "promoted_sota_chase_rows": sum(1 for row in promoted if row.get("promotion_track") == "sota_chase"),
        "unsafe_promoted_rows": len(unsafe),
        "blocked_rows_by_reason": dict(sorted(blocked_reasons.items())),
    }
