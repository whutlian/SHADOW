from __future__ import annotations

from shadow_hgc.sft.ttcpp_ratio_curve import aggregate_ratio_curve, make_ratio_curve_row


def test_t33_ratio_curve_row_accounts_strict_ratio_and_mixup() -> None:
    row = make_ratio_curve_row(
        dataset="Reddit",
        method="reddit_ttcpp_gamlp_table_student",
        seed=42,
        ratio=0.001,
        accuracy=0.93,
        macro_f1=0.89,
        virtual_mixup_count=128,
    )
    assert row["total_condensed_nodes"] == 233
    assert abs(row["actual_full_node_ratio"] - 233 / 232_965) < 1e-12
    assert row["virtual_mixup_count"] == 128


def test_t33_multiseed_aggregation_reports_mean_std_gap_and_best() -> None:
    rows = [
        {"dataset": "Reddit", "method": "m", "requested_full_node_ratio": 0.001, "seed": 1, "accuracy": 0.92, "macro_f1": 0.88, "valid_acc": 0.91},
        {"dataset": "Reddit", "method": "m", "requested_full_node_ratio": 0.001, "seed": 2, "accuracy": 0.94, "macro_f1": 0.90, "valid_acc": 0.93},
    ]
    agg = aggregate_ratio_curve(rows)
    assert len(agg) == 1
    row = agg[0]
    assert row["seed_count"] == 2
    assert abs(row["accuracy_mean"] - 0.93) < 1e-12
    assert row["accuracy_std"] > 0.0
    assert row["accuracy_best"] == 0.94
    assert abs(row["valid_test_gap_mean"] - 0.01) < 1e-12
