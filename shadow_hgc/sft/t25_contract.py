from __future__ import annotations

from typing import Any

from shadow_hgc.ratio.scale_bucket import account_full_node_ratio


T25_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_target_cache",
    "uses_exact_pairwise",
    "full_class_kmeans",
)


T25_OUTPUT_FIELDS: list[str] = [
    "dataset",
    "method",
    "seed",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "target_prototypes",
    "shadow_nodes",
    "total_condensed_nodes",
    "total_condensed_edges",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "predicted_class_count",
    "precompute_time",
    "condensation_time",
    "training_time",
    "inference_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "full_edge_scans",
    "hnr_edge_scans",
    "hnr_cache_bytes",
    "fdm_signature_dim",
    "fdm_num_subclasses",
    "fdm_candidate_pool_size",
    "fdm_mode",
    "hnr_hist_mode",
    "shadow_b",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_target_cache",
    "uses_exact_pairwise",
    "full_class_kmeans",
    "status",
    "promoted",
    "promotion_status",
    "failure_reason",
    "notes",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def validate_t25_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    if str(row.get("promotion_status", "")) == "promoted" or _truthy(row.get("promoted", False)):
        for flag in T25_FORBIDDEN_PROMOTED_FLAGS:
            if _truthy(row.get(flag, False)):
                forbidden.append(flag)
        if row.get("ratio_mode") != "full_node":
            forbidden.append("ratio_mode_not_full_node")
        if row.get("actual_full_node_ratio", "") in {"", None}:
            forbidden.append("missing_full_node_ratio")
        if row.get("accuracy", "") in {"", None}:
            forbidden.append("missing_accuracy_for_promotion")
        if row.get("macro_f1", "") in {"", None}:
            forbidden.append("missing_macro_f1_for_promotion")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def apply_ultra_safe_guards(options: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(options)
    guarded["ultra_safe"] = True
    guarded["fdm_mode"] = "lite"
    if guarded.get("hnr_hist_mode") == "full":
        guarded["hnr_hist_mode"] = "topk"
    guarded.setdefault("hnr_hist_mode", "topk")
    for key in (
        "uses_all_target_cache",
        "uses_exact_pairwise",
        "full_class_kmeans",
        "uses_dense_p2",
        "uses_e_by_d_materialization",
        "uses_full_edge_index_on_gpu",
    ):
        guarded[key] = False
    guarded["train_target_only_cache"] = True
    return guarded


def make_t25_row(
    *,
    dataset: str,
    method: str,
    requested_full_node_ratio: float,
    original_total_nodes: int,
    target_prototypes: int,
    shadow_nodes: int,
    total_condensed_edges: int,
    seed: int = 42,
    accuracy: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
    status: str = "ready",
    promotion_status: str = "not_promoted",
    failure_reason: str = "",
    notes: str = "",
    **extra: Any,
) -> dict[str, Any]:
    accounting = account_full_node_ratio(
        original_total_nodes=int(original_total_nodes),
        target_prototypes=int(target_prototypes),
        shadow_nodes=int(shadow_nodes),
        condensed_edges=int(total_condensed_edges),
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "ratio_mode": "full_node",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "predicted_class_count": predicted_classes,
        "precompute_time": "",
        "condensation_time": "",
        "training_time": "",
        "inference_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "full_edge_scans": "",
        "hnr_edge_scans": 0,
        "hnr_cache_bytes": 0,
        "fdm_signature_dim": "",
        "fdm_num_subclasses": "",
        "fdm_candidate_pool_size": "",
        "fdm_mode": "",
        "hnr_hist_mode": "",
        "shadow_b": "",
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_all_target_cache": False,
        "uses_exact_pairwise": False,
        "full_class_kmeans": False,
        "status": status,
        "promoted": promotion_status == "promoted",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": notes,
        **accounting,
    }
    row["requested_full_node_ratio"] = float(requested_full_node_ratio)
    row["total_condensed_edges"] = int(total_condensed_edges)
    row.update(extra)
    safety = validate_t25_promoted_row(row)
    if not safety["valid"]:
        row["promoted"] = False
        row["promotion_status"] = "blocked_forbidden"
        row["failure_reason"] = ",".join(safety["forbidden_flags"])
    return row
