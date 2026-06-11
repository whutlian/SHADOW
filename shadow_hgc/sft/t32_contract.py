from __future__ import annotations

from collections import Counter
from typing import Any


T32_STAGE = "t32"

REDDIT_NUM_NODES = 232_965
ARXIV_NUM_NODES = 169_343
PRODUCTS_NUM_NODES = 2_449_029

REDDIT_NUM_CLASSES = 41
ARXIV_NUM_CLASSES = 40
PRODUCTS_NUM_CLASSES = 47

T32_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "method",
    "stage",
    "seed",
    "promotion_track",
    "promotion_status",
    "promotion_allowed",
    "status",
    "failure_reason",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "original_num_nodes",
    "condensed_nodes",
    "shadow_nodes",
    "condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_acc",
    "predicted_classes",
    "teacher_accuracy",
    "teacher_valid_acc",
    "teacher_temperature",
    "teacher_entropy_mean",
    "teacher_disagreement_mean",
    "uses_teacher_logits",
    "uses_kd",
    "uses_logits_as_input",
    "uses_external_text_features",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "candidate_nodes",
    "student_model",
    "hidden_dim",
    "epochs",
    "dropout",
    "weight_decay",
    "lambda_soft",
    "lambda_hard",
    "lambda_prior",
    "lambda_conf",
    "lambda_mix",
    "soft_temperature",
    "budget_policy",
    "row_type_counts_json",
    "selected_soft_prior_kl",
    "entropy_bucket_coverage",
    "margin_bucket_coverage",
    "disagreement_bucket_coverage",
    "class_coverage_min",
    "class_coverage_median",
    "class_coverage_max",
    "cache_bytes",
    "precompute_time",
    "selection_time",
    "training_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "notes",
    "base_predictor",
    "base_logit_cache_path",
    "base_accuracy",
    "base_valid_acc",
    "cns_accuracy",
    "cns_valid_acc",
    "graph_direction",
    "correction_alpha",
    "smoothing_alpha",
    "correction_steps",
    "smoothing_steps",
    "autoscale",
    "normalization_mode",
    "self_loop_mode",
    "split_hash",
    "feature_manifest_hash",
    "semantic_encoder",
    "semantic_cache_path",
    "semantic_dim",
    "raw_text_map_path",
    "node_id_to_paper_id_path",
    "semantic_cache_bytes",
    "semantic_encode_time",
    "uses_temporal_features",
    "temporal_decay_gamma",
    "next_action",
]

SAFE_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_teacher_logits",
    "uses_kd",
    "uses_logits_as_input",
    "uses_external_text_features",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
)

SOTA_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_dense_adjacency",
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
    raise ValueError(f"unknown T32 dataset: {dataset}")


def ratio_budget(dataset: str, requested_full_node_ratio: float) -> int:
    return max(1, int(round(original_num_nodes(dataset) * float(requested_full_node_ratio))))


def default_flags() -> dict[str, bool]:
    return {
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_logits_as_input": False,
        "uses_external_text_features": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_adjacency": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
    }


def make_t32_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    requested_full_node_ratio: float = 0.0,
    condensed_nodes: int | None = None,
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
    notes: str = "",
    next_action: str = "",
    extra: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    nodes = original_num_nodes(dataset)
    if condensed_nodes is None:
        condensed_nodes = ratio_budget(dataset, requested_full_node_ratio) if requested_full_node_ratio else 0
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "stage": T32_STAGE,
        "seed": int(seed),
        "promotion_track": promotion_track,
        "promotion_status": promotion_status,
        "promotion_allowed": promotion_status == "promoted",
        "status": status,
        "failure_reason": failure_reason,
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "actual_full_node_ratio": float(condensed_nodes) / float(nodes) if nodes else 0.0,
        "original_num_nodes": nodes,
        "condensed_nodes": int(condensed_nodes),
        "shadow_nodes": int(shadow_nodes),
        "condensed_edges": int(condensed_edges),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_acc": valid_acc,
        "predicted_classes": predicted_classes,
        "teacher_accuracy": "",
        "teacher_valid_acc": "",
        "teacher_temperature": "",
        "teacher_entropy_mean": "",
        "teacher_disagreement_mean": "",
        "candidate_nodes": "",
        "student_model": "",
        "hidden_dim": "",
        "epochs": "",
        "dropout": "",
        "weight_decay": "",
        "lambda_soft": 1.0,
        "lambda_hard": "",
        "lambda_prior": "",
        "lambda_conf": "",
        "lambda_mix": "",
        "soft_temperature": "",
        "budget_policy": "",
        "row_type_counts_json": "",
        "selected_soft_prior_kl": "",
        "entropy_bucket_coverage": "",
        "margin_bucket_coverage": "",
        "disagreement_bucket_coverage": "",
        "class_coverage_min": "",
        "class_coverage_median": "",
        "class_coverage_max": "",
        "cache_bytes": "",
        "precompute_time": "",
        "selection_time": "",
        "training_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "notes": notes,
        "base_predictor": "",
        "base_logit_cache_path": "",
        "base_accuracy": "",
        "base_valid_acc": "",
        "cns_accuracy": "",
        "cns_valid_acc": "",
        "graph_direction": "",
        "correction_alpha": "",
        "smoothing_alpha": "",
        "correction_steps": "",
        "smoothing_steps": "",
        "autoscale": "",
        "normalization_mode": "",
        "self_loop_mode": "",
        "split_hash": "",
        "feature_manifest_hash": "",
        "semantic_encoder": "",
        "semantic_cache_path": "",
        "semantic_dim": "",
        "raw_text_map_path": "",
        "node_id_to_paper_id_path": "",
        "semantic_cache_bytes": "",
        "semantic_encode_time": "",
        "uses_temporal_features": False,
        "temporal_decay_gamma": "",
        "next_action": next_action,
        **default_flags(),
    }
    row.update(fields)
    if extra:
        row.update(extra)
    for field in T32_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def wants_promotion(row: dict[str, Any]) -> bool:
    return row.get("promotion_status") == "promoted" or truthy(row.get("promotion_allowed", False))


def validate_t32_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    forbidden.extend(f"missing_field:{field}" for field in T32_REQUIRED_FIELDS if field not in row)
    track = str(row.get("promotion_track", "safe_main"))
    flags = SOTA_FORBIDDEN_FLAGS if track == "sota_chase" else SAFE_FORBIDDEN_FLAGS
    for flag in flags:
        if truthy(row.get(flag, False)):
            forbidden.append(flag)
    if truthy(row.get("uses_teacher_logits", False)) and track != "sota_chase":
        forbidden.append("uses_teacher_logits_requires_sota_chase")
    if wants_promotion(row):
        if row.get("accuracy") in {"", None}:
            forbidden.append("missing_accuracy")
        if row.get("macro_f1") in {"", None}:
            forbidden.append("missing_macro_f1")
        status = str(row.get("status", ""))
        if status in {"blocked", "failed"} or "smoke" in status or "carried_forward" in status:
            forbidden.append("status_not_promotable")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_t32_promotion_guard(row: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(row)
    if not wants_promotion(guarded):
        guarded["promotion_status"] = guarded.get("promotion_status") or "not_promoted"
        guarded["promotion_allowed"] = False
        return guarded
    result = validate_t32_row(guarded)
    if not result["valid"]:
        guarded["promotion_status"] = "blocked_forbidden"
        guarded["promotion_allowed"] = False
        guarded["failure_reason"] = ",".join(result["forbidden_flags"])
        return guarded
    guarded["promotion_status"] = "promoted"
    guarded["promotion_allowed"] = True
    return guarded


def ttcpp_promotion_status(*, ratio: float, accuracy: float, macro_f1: float) -> tuple[str, str]:
    ratio = float(ratio)
    acc = float(accuracy)
    macro = float(macro_f1)
    if abs(ratio - 0.001) < 1e-12:
        if acc < 0.923:
            return "not_promoted", "ttcpp_accuracy_gate_not_met"
        if macro < 0.885:
            return "not_promoted", "ttcpp_macro_gate_not_met"
        return "promoted", ""
    if abs(ratio - 0.005) < 1e-12:
        if acc < 0.938:
            return "not_promoted", "ttcpp_accuracy_gate_not_met"
        if macro < 0.906:
            return "not_promoted", "ttcpp_macro_gate_not_met"
        return "promoted", ""
    if abs(ratio - 0.01) < 1e-12:
        return ("promoted", "") if acc >= 0.940 else ("not_promoted", "ttcpp_accuracy_gate_not_met")
    return "not_promoted", "ttcpp_ratio_gate_not_defined"


def summarize_guard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if wants_promotion(row)]
    unsafe = [row for row in promoted if not validate_t32_row(row)["valid"]]
    blocked = Counter(str(row.get("failure_reason", "")) for row in rows if str(row.get("failure_reason", "")))
    return {
        "rows": len(rows),
        "promoted_safe_rows": sum(1 for row in promoted if row.get("promotion_track") == "safe_main"),
        "promoted_sota_chase_rows": sum(1 for row in promoted if row.get("promotion_track") == "sota_chase"),
        "unsafe_promoted_rows": len(unsafe),
        "blocked_rows_by_reason": dict(sorted(blocked.items())),
    }
