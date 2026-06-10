from __future__ import annotations

from typing import Any

from shadow_hgc.ratio.scale_bucket import account_full_node_ratio
from shadow_hgc.sft.t25_contract import T25_FORBIDDEN_PROMOTED_FLAGS


T26_STAGE = "t26"

T26_PRODUCTS_METHODS: tuple[str, ...] = (
    "products_cb_random",
    "products_cb_kcenter",
    "products_cb_herding",
    "products_cb_hybrid",
    "products_uca_kmeans_labeled_nearest",
    "products_uca_hybrid",
    "products_uca_hybrid_mixup",
    "products_uca_hybrid_balanced_trainer",
)

T26_PRODUCTS_DIAGNOSTICS: tuple[str, ...] = (
    "P0a_alltrain_condensed_trainer_parity",
    "P0b_selected_prototype_self_fit",
    "P0c_same_budget_random_subset",
    "P0d_nearest_prototype_oracle",
    "P0e_per_class_collapse_report",
    "P0f_feature_normalization_parity",
)

T26_REDDIT_METHODS: tuple[str, ...] = (
    "reddit_current_sft_signature_random",
    "reddit_current_sft_signature_medoid",
    "reddit_current_sft_signature_kcenter",
    "reddit_sft_hnr_fdm_hybrid",
    "reddit_tuned_balanced_trainer",
    "reddit_sft_signature_mixup",
    "reddit_true_shadow_b1",
)

T26_ARXIV_METHODS: tuple[str, ...] = (
    "arxiv_teacher_sft_v4",
    "arxiv_teacher_sagn_lite",
    "arxiv_condensation_blocked_until_teacher_gate",
)

T26_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *T25_FORBIDDEN_PROMOTED_FLAGS,
            "uses_logits",
            "uses_legacy_diffusion",
            "uses_old_diffusion",
            "uses_full_edge_backprop",
            "uses_source_anchors",
            "uses_new_exposed_schema",
            "uses_dense_synthetic_adjacency",
            "uses_valid_test_labels_for_selection",
            "uca_uses_valid_test_labels",
            "is_proxy",
        )
    )
)

T26_REQUIRED_FIELDS: list[str] = [
    "dataset",
    "stage",
    "method",
    "seed",
    "split",
    "ratio_mode",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "target_prototypes",
    "shadow_nodes",
    "other_condensed_nodes",
    "total_condensed_nodes",
    "total_condensed_edges",
    "accuracy",
    "macro_f1",
    "valid_accuracy",
    "valid_macro_f1",
    "predicted_classes",
    "predicted_class_count",
    "per_class_report_path",
    "precompute_time",
    "condensation_time",
    "training_time",
    "inference_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "disk_bytes",
    "full_edge_scans",
    "edge_slice_cache_bytes",
    "sft_manifest_dir",
    "sft_manifest_hash",
    "signature_cache_dir",
    "signature_hash",
    "signature_dim",
    "signature_blocks",
    "class_budget_policy",
    "class_budget_floor",
    "class_budget_min",
    "class_budget_max",
    "class_budget_json",
    "trainer_recipe",
    "trainer_balanced_batches",
    "trainer_label_smoothing",
    "trainer_mixup_alpha",
    "trainer_ema",
    "trainer_swa",
    "trainer_logit_adjustment",
    "selection_score",
    "coverage_gap_l1",
    "coverage_gap_l2",
    "domains_total",
    "domains_without_train_support",
    "domains_without_unlabeled_support",
    "uca_num_domains",
    "uca_domain_seed",
    "uca_uses_valid_test_labels",
    "p0a_alltrain_acc",
    "p0a_passed",
    "p0b_self_fit_acc",
    "p0b_passed",
    "p0c_random_subset_acc",
    "p0d_prototype_oracle_acc",
    "p0d_centroid_oracle_acc",
    "p0e_predicted_class_collapse",
    "p0f_normalization_parity",
    "shadow_graph_materialized",
    "shadow_b",
    "shadow_edge_weight_nonnegative",
    "shadow_exposed_schema_preserved",
    "uses_logits_as_input",
    "uses_logits",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_legacy_diffusion",
    "uses_old_diffusion",
    "uses_full_edge_backprop",
    "uses_e_by_d_materialization",
    "uses_full_edge_index_on_gpu",
    "uses_all_target_cache",
    "uses_exact_pairwise",
    "uses_source_anchors",
    "uses_new_exposed_schema",
    "uses_dense_synthetic_adjacency",
    "full_class_kmeans",
    "is_proxy",
    "status",
    "promoted",
    "promotion_status",
    "promotion_reason",
    "failure_reason",
    "notes",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def validate_t26_promoted_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden: list[str] = []
    wants_promotion = str(row.get("promotion_status", "")) == "promoted" or _truthy(row.get("promoted", False))
    for flag in T26_FORBIDDEN_PROMOTED_FLAGS:
        if _truthy(row.get(flag, False)):
            forbidden.append(flag)
    if wants_promotion:
        if row.get("ratio_mode") != "full_node":
            forbidden.append("ratio_mode_not_full_node")
        if row.get("actual_full_node_ratio", "") in {"", None}:
            forbidden.append("missing_full_node_ratio")
        if row.get("accuracy", "") in {"", None}:
            forbidden.append("missing_accuracy_for_promotion")
        if row.get("macro_f1", "") in {"", None}:
            forbidden.append("missing_macro_f1_for_promotion")
        dataset = str(row.get("dataset", ""))
        predicted = row.get("predicted_class_count", row.get("predicted_classes", ""))
        if dataset == "ogbn-products" and predicted not in {"", None} and int(float(predicted)) < 45:
            forbidden.append("products_predicted_class_count_below_45")
        if "shadow" in str(row.get("method", "")) and not _truthy(row.get("shadow_graph_materialized", False)):
            forbidden.append("shadow_graph_not_materialized")
    return {"valid": not forbidden, "forbidden_flags": forbidden}


def default_t26_flags() -> dict[str, Any]:
    return {
        "uses_logits_as_input": False,
        "uses_logits": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_legacy_diffusion": False,
        "uses_old_diffusion": False,
        "uses_full_edge_backprop": False,
        "uses_e_by_d_materialization": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_all_target_cache": False,
        "uses_exact_pairwise": False,
        "uses_source_anchors": False,
        "uses_new_exposed_schema": False,
        "uses_dense_synthetic_adjacency": False,
        "full_class_kmeans": False,
        "uca_uses_valid_test_labels": False,
        "is_proxy": False,
        "shadow_graph_materialized": False,
        "shadow_edge_weight_nonnegative": True,
        "shadow_exposed_schema_preserved": True,
    }


def make_t26_row(
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
    promotion_reason: str = "",
    failure_reason: str = "",
    notes: str = "",
    split: str = "",
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
        "stage": T26_STAGE,
        "method": method,
        "seed": int(seed),
        "split": split,
        "ratio_mode": "full_node",
        "requested_full_node_ratio": float(requested_full_node_ratio),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "valid_accuracy": "",
        "valid_macro_f1": "",
        "predicted_classes": predicted_classes,
        "predicted_class_count": predicted_classes,
        "per_class_report_path": "",
        "precompute_time": "",
        "condensation_time": "",
        "training_time": "",
        "inference_time": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "cache_bytes": "",
        "disk_bytes": "",
        "full_edge_scans": "",
        "edge_slice_cache_bytes": "",
        "sft_manifest_dir": "",
        "sft_manifest_hash": "",
        "signature_cache_dir": "",
        "signature_hash": "",
        "signature_dim": "",
        "signature_blocks": "",
        "class_budget_policy": "",
        "class_budget_floor": "",
        "class_budget_min": "",
        "class_budget_max": "",
        "class_budget_json": "",
        "trainer_recipe": "",
        "trainer_balanced_batches": False,
        "trainer_label_smoothing": "",
        "trainer_mixup_alpha": "",
        "trainer_ema": False,
        "trainer_swa": False,
        "trainer_logit_adjustment": False,
        "selection_score": "",
        "coverage_gap_l1": "",
        "coverage_gap_l2": "",
        "domains_total": "",
        "domains_without_train_support": "",
        "domains_without_unlabeled_support": "",
        "uca_num_domains": "",
        "uca_domain_seed": "",
        "p0a_alltrain_acc": "",
        "p0a_passed": "",
        "p0b_self_fit_acc": "",
        "p0b_passed": "",
        "p0c_random_subset_acc": "",
        "p0d_prototype_oracle_acc": "",
        "p0d_centroid_oracle_acc": "",
        "p0e_predicted_class_collapse": "",
        "p0f_normalization_parity": "",
        "shadow_b": "",
        **default_t26_flags(),
        "status": status,
        "promoted": promotion_status == "promoted",
        "promotion_status": promotion_status,
        "promotion_reason": promotion_reason,
        "failure_reason": failure_reason,
        "notes": notes,
        **accounting,
    }
    row["requested_full_node_ratio"] = float(requested_full_node_ratio)
    row["total_condensed_edges"] = int(total_condensed_edges)
    row.update(extra)
    safety = validate_t26_promoted_row(row)
    if not safety["valid"]:
        row["promoted"] = False
        row["promotion_status"] = "blocked_forbidden"
        row["failure_reason"] = ",".join(safety["forbidden_flags"])
    return row


def summarize_requirement_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in rows if str(row.get("promotion_status", "")) == "promoted"]
    forbidden_promoted = [row for row in promoted if not validate_t26_promoted_row(row)["valid"]]
    return {
        "rows": int(len(rows)),
        "promoted_rows": int(len(promoted)),
        "forbidden_promoted_rows": int(len(forbidden_promoted)),
        "has_performance_regression": any(str(row.get("failure_reason", "")).startswith("no_regression") for row in rows),
        "all_promoted_safe": not forbidden_promoted,
    }
