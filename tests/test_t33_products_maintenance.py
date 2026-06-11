from __future__ import annotations

from scripts.run_t33_products_maintenance import build_products_rows
from shadow_hgc.sft.t33_products import aggregate_products_maintenance


class Args:
    seed = 42
    methods = ["products_uca_hybrid_mixup"]
    ratios = [0.0002]
    seeds = [1, 42]


def test_t33_products_carried_forward_rows_are_not_promoted() -> None:
    rows = build_products_rows(Args())
    seed42 = [row for row in rows if int(row["seed"]) == 42][0]
    seed1 = [row for row in rows if int(row["seed"]) == 1][0]
    assert seed42["status"] == "carried_forward_reference"
    assert seed42["promotion_status"] == "not_promoted"
    assert seed1["status"] == "blocked"
    assert seed1["failure_reason"] == "missing_products_seed_reference"


def test_t33_products_aggregate_reports_mean_std_and_class_range() -> None:
    rows = [
        {"method": "m", "requested_full_node_ratio": 0.0002, "accuracy": 0.7, "macro_f1": 0.3, "predicted_classes": 20},
        {"method": "m", "requested_full_node_ratio": 0.0002, "accuracy": 0.8, "macro_f1": 0.4, "predicted_classes": 30},
    ]
    agg = aggregate_products_maintenance(rows)
    assert agg[0]["accuracy_mean"] == 0.75
    assert agg[0]["predicted_classes_min"] == 20
    assert agg[0]["predicted_classes_max"] == 30
