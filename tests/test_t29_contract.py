from __future__ import annotations

from shadow_hgc.sft.t29_contract import (
    T29_REQUIRED_FIELDS,
    apply_t29_promotion_guard,
    make_t29_row,
    ratio_budget,
    validate_t29_row,
)


def test_t29_row_schema_required_fields():
    row = make_t29_row(
        dataset="Reddit",
        method="reddit_sft_omcp_random",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=1,
        accuracy=0.924,
        macro_f1=0.887,
        predicted_classes=41,
        status="completed_long",
        promotion_status="promoted",
        promotion_track="safe_mainline",
    )
    assert set(T29_REQUIRED_FIELDS).issubset(row)
    assert row["actual_condensed_nodes"] == 1
    assert row["actual_full_node_ratio"] == 0.001


def test_t29_safe_promotion_forbidden_flags():
    row = make_t29_row(
        dataset="Reddit",
        method="reddit_sft_omcp_random",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=1,
        accuracy=0.924,
        macro_f1=0.887,
        predicted_classes=41,
        status="completed_long",
        promotion_status="promoted",
        promotion_track="safe_mainline",
        uses_teacher_logits=True,
    )
    result = validate_t29_row(row)
    assert result["valid"] is False
    assert "uses_teacher_logits" in result["forbidden_flags"]
    guarded = apply_t29_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"


def test_t29_sota_chase_allows_teacher_logits_but_not_test_labels():
    allowed = make_t29_row(
        dataset="Reddit",
        method="reddit_pltc_omcp",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=1,
        accuracy=0.927,
        macro_f1=0.889,
        predicted_classes=41,
        status="completed_long",
        promotion_status="promoted",
        promotion_track="sota_chase",
        uses_teacher_logits=True,
    )
    assert validate_t29_row(allowed)["valid"] is True

    blocked = dict(allowed)
    blocked["uses_test_labels_as_input"] = True
    result = validate_t29_row(blocked)
    assert result["valid"] is False
    assert "uses_test_labels_as_input" in result["forbidden_flags"]


def test_t29_ratio_accounting_full_node_and_budget_scales():
    low = ratio_budget("Reddit", 0.001)
    high = ratio_budget("Reddit", 0.005)
    assert low == 233
    assert high == 1165
    assert low != high

    row = make_t29_row(
        dataset="Reddit",
        method="reddit_sft_omcp_random",
        seed=42,
        requested_full_node_ratio=0.005,
        target_prototypes=high,
        original_num_nodes=232_965,
        status="completed_operator_smoke",
    )
    assert abs(row["actual_full_node_ratio"] - 0.005) <= max(1 / 232_965, 0.05 * 0.005)


def test_t29_structure_budget_bug_rejects_fixed_96_nodes_for_different_ratios():
    low = make_t29_row(
        dataset="Reddit",
        method="reddit_sft_omcp_random",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=232_965,
        target_prototypes=96,
        status="completed_operator_smoke",
        promotion_status="promoted",
    )
    high = make_t29_row(
        dataset="Reddit",
        method="reddit_sft_omcp_random",
        seed=42,
        requested_full_node_ratio=0.005,
        original_num_nodes=232_965,
        target_prototypes=96,
        status="completed_operator_smoke",
        promotion_status="promoted",
    )
    assert validate_t29_row(low)["valid"] is False
    assert validate_t29_row(high)["valid"] is False
