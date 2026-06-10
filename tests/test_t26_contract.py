import pytest

from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row, validate_t26_promoted_row


def test_t26_full_node_ratio_counts_target_and_shadow_only():
    row = make_t26_row(
        dataset="Reddit",
        method="reddit_tuned",
        requested_full_node_ratio=0.005,
        original_total_nodes=1000,
        target_prototypes=3,
        shadow_nodes=2,
        total_condensed_edges=7,
        accuracy=0.93,
        macro_f1=0.89,
    )

    assert row["actual_full_node_ratio"] == pytest.approx(0.005)
    assert row["total_condensed_nodes"] == 5
    assert row["ratio_mode"] == "full_node"


def test_t26_promoted_row_blocks_forbidden_components():
    row = make_t26_row(
        dataset="ogbn-products",
        method="products_uca_hybrid",
        requested_full_node_ratio=0.0025,
        original_total_nodes=1000,
        target_prototypes=3,
        shadow_nodes=0,
        total_condensed_edges=3,
        accuracy=0.75,
        macro_f1=0.40,
        promotion_status="promoted",
        uses_kd=True,
    )

    assert row["promotion_status"] == "blocked_forbidden"
    assert "uses_kd" in row["failure_reason"]
    assert validate_t26_promoted_row(row)["valid"] is False


def test_t26_required_fields_cover_stage_outputs():
    for field in [
        "dataset",
        "stage",
        "method",
        "coverage_gap_l1",
        "p0a_passed",
        "uca_uses_valid_test_labels",
        "promotion_status",
        "failure_reason",
    ]:
        assert field in T26_REQUIRED_FIELDS
