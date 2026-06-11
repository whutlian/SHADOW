from __future__ import annotations

from shadow_hgc.sft.t31_contract import (
    T31_REQUIRED_FIELDS,
    apply_t31_promotion_guard,
    make_t31_row,
    ratio_budget,
    validate_t31_row,
)


def test_t31_schema_contains_stage_specific_fields() -> None:
    required = set(T31_REQUIRED_FIELDS)
    for field in [
        "syn_rows",
        "shadow_nodes",
        "condensed_edges",
        "teacher_method",
        "teacher_entropy_mean",
        "candidate_bucket_counts_json",
        "soft_class_mass_coverage",
        "uses_valid_labels_for_hyperparam_selection",
        "semantic_match_rate",
        "graph_direction",
    ]:
        assert field in required


def test_t31_ratio_budget_uses_full_node_counts() -> None:
    assert ratio_budget("Reddit", 0.001) == 233
    assert ratio_budget("Reddit", 0.005) == 1165
    assert ratio_budget("ogbn-arxiv", 0.005) == 847


def test_t31_safe_main_rejects_teacher_logits_and_kd() -> None:
    row = make_t31_row(
        dataset="Reddit",
        method="reddit_ttc_confidence_balanced",
        seed=42,
        requested_full_node_ratio=0.001,
        total_condensed_nodes=233,
        accuracy=0.93,
        macro_f1=0.89,
        predicted_classes=41,
        status="completed_long",
        promotion_track="safe_main",
        promotion_status="promoted",
        uses_teacher_logits=True,
    )
    guarded = apply_t31_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_logits" in guarded["failure_reason"]


def test_t31_sota_allows_teacher_logits_but_rejects_valid_test_inputs() -> None:
    row = make_t31_row(
        dataset="Reddit",
        method="reddit_ttc_confidence_balanced",
        seed=42,
        requested_full_node_ratio=0.005,
        total_condensed_nodes=1165,
        accuracy=0.934,
        macro_f1=0.893,
        predicted_classes=41,
        status="completed_long",
        promotion_track="sota_chase",
        promotion_status="promoted",
        uses_teacher_logits=True,
        uses_valid_labels_as_input=True,
    )
    result = validate_t31_row(row)
    assert not result["valid"]
    assert "uses_valid_labels_as_input" in result["forbidden_flags"]


def test_t31_promoted_rows_require_real_metrics_and_nonblocked_status() -> None:
    row = make_t31_row(
        dataset="ogbn-arxiv",
        method="arxiv_semantic_sft",
        seed=42,
        status="blocked",
        failure_reason="raw_text_or_semantic_cache_missing",
        promotion_track="sota_chase",
        promotion_status="promoted",
        uses_external_text_features=True,
    )
    guarded = apply_t31_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "missing_accuracy" in guarded["failure_reason"]
    assert "status_not_promotable" in guarded["failure_reason"]
