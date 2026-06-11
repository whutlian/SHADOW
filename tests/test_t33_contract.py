from __future__ import annotations

from shadow_hgc.sft.t33_contract import (
    T33_REQUIRED_FIELDS,
    apply_t33_promotion_guard,
    make_t33_row,
    ratio_budget,
    reddit_gate_status,
    validate_t33_row,
)


def test_t33_schema_contains_scale_and_cache_fields() -> None:
    required = set(T33_REQUIRED_FIELDS)
    for field in [
        "ratio_mode",
        "total_condensed_nodes",
        "teacher_cache_mode",
        "uses_teacher_probs",
        "uses_all_pair_distance",
        "selected_soft_prior_kl_to_teacher_prior",
        "teacher_pairwise_kl_mean",
        "logits_cache_hash",
    ]:
        assert field in required


def test_t33_ratio_budget_uses_strict_full_node_count() -> None:
    assert ratio_budget("Reddit", 0.0005) == 116
    assert ratio_budget("Reddit", 0.001) == 233
    assert ratio_budget("ogbn-products", 0.0002) == 490


def test_t33_safe_main_rejects_teacher_probs_and_sota_requires_cache_diag() -> None:
    safe = make_t33_row(
        dataset="Reddit",
        method="reddit_ttcpp_ratio_adaptive_v2",
        seed=42,
        requested_full_node_ratio=0.001,
        accuracy=0.93,
        macro_f1=0.89,
        status="completed_long",
        promotion_track="safe_main",
        promotion_status="promoted",
        uses_teacher_probs=True,
    )
    guarded = apply_t33_promotion_guard(safe)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_probs" in guarded["failure_reason"]

    sota = make_t33_row(
        dataset="Reddit",
        method="reddit_ttcpp_gamlp_table_student",
        seed=42,
        requested_full_node_ratio=0.001,
        accuracy=0.93,
        macro_f1=0.89,
        status="completed_long",
        promotion_track="sota_chase",
        promotion_status="promoted",
        uses_teacher_probs=True,
        uses_teacher_logits=True,
        uses_logits_as_input=False,
    )
    result = validate_t33_row(sota)
    assert not result["valid"]
    assert "missing_teacher_cache_mode" in result["forbidden_flags"]
    assert "missing_teacher_cache_bytes" in result["forbidden_flags"]


def test_t33_ultra_promotion_rejects_dense_teacher_cache() -> None:
    row = make_t33_row(
        dataset="ogbn-papers100M",
        method="ttcpp_ultra_topk_cache_planner_papers100M",
        seed=0,
        requested_full_node_ratio=0.0001,
        status="completed_dry_run",
        promotion_track="ultra_planner",
        promotion_status="promoted",
        teacher_cache_mode="dense_fp16",
        uses_dense_teacher_cache_in_ram=True,
    )
    guarded = apply_t33_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "ultra_dense_teacher_cache" in guarded["failure_reason"]


def test_t33_reddit_gates_are_ratio_specific_and_macro_guarded() -> None:
    assert reddit_gate_status(ratio=0.001, accuracy=0.9306, macro_f1=0.895) == ("promoted", "")
    assert reddit_gate_status(ratio=0.005, accuracy=0.9382, macro_f1=0.9039) == (
        "not_promoted",
        "ttcpp_macro_regression",
    )
    assert reddit_gate_status(ratio=0.005, accuracy=0.9379, macro_f1=0.906) == (
        "not_promoted",
        "ttcpp_accuracy_gate_not_met",
    )
