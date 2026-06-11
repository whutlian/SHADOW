from __future__ import annotations

from shadow_hgc.sft.t32_contract import (
    T32_REQUIRED_FIELDS,
    apply_t32_promotion_guard,
    make_t32_row,
    ratio_budget,
    ttcpp_promotion_status,
    validate_t32_row,
)


def test_t32_schema_contains_stage_fields() -> None:
    required = set(T32_REQUIRED_FIELDS)
    for field in [
        "condensed_nodes",
        "uses_logits_as_input",
        "lambda_conf",
        "budget_policy",
        "selected_soft_prior_kl",
        "base_logit_cache_path",
        "normalization_mode",
        "semantic_encoder",
    ]:
        assert field in required


def test_t32_ratio_budget_uses_full_node_counts() -> None:
    assert ratio_budget("Reddit", 0.001) == 233
    assert ratio_budget("Reddit", 0.005) == 1165
    assert ratio_budget("ogbn-arxiv", 0.005) == 847


def test_t32_ttc_teacher_logits_require_sota_track() -> None:
    row = make_t32_row(
        dataset="Reddit",
        method="reddit_ttcpp_ratio_adaptive_core70",
        seed=42,
        requested_full_node_ratio=0.005,
        condensed_nodes=1165,
        accuracy=0.94,
        macro_f1=0.907,
        status="completed_long",
        promotion_track="safe_main",
        promotion_status="promoted",
        uses_teacher_logits=True,
    )
    guarded = apply_t32_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_logits" in guarded["failure_reason"]


def test_t32_promoted_rows_require_real_metrics_and_nonblocked_status() -> None:
    row = make_t32_row(
        dataset="Reddit",
        method="reddit_ttcpp_ratio_adaptive_core40",
        seed=42,
        status="blocked",
        promotion_track="sota_chase",
        promotion_status="promoted",
        uses_teacher_logits=True,
    )
    result = validate_t32_row(row)
    assert not result["valid"]
    assert "missing_accuracy" in result["forbidden_flags"]
    assert "status_not_promotable" in result["forbidden_flags"]


def test_t32_ttcpp_gate_requires_ratio_specific_accuracy_and_macro() -> None:
    assert ttcpp_promotion_status(ratio=0.005, accuracy=0.939, macro_f1=0.907) == ("promoted", "")
    assert ttcpp_promotion_status(ratio=0.005, accuracy=0.939, macro_f1=0.905) == (
        "not_promoted",
        "ttcpp_macro_gate_not_met",
    )
    assert ttcpp_promotion_status(ratio=0.001, accuracy=0.9229, macro_f1=0.89) == (
        "not_promoted",
        "ttcpp_accuracy_gate_not_met",
    )
