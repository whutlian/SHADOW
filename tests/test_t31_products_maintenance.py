from __future__ import annotations

from argparse import Namespace

from scripts.run_t31_products_maintenance import build_products_rows


def test_t31_products_maintenance_reports_seed42_and_missing_other_seeds() -> None:
    rows = build_products_rows(Namespace(seed=42, seeds=[1, 42], ratios=[0.0002], methods=["products_uca_hybrid_mixup"]))
    assert len(rows) == 2
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "missing_products_seed_reference"
    assert rows[1]["status"] == "carried_forward_reference"
    assert rows[1]["accuracy"] == 0.6858000868
    assert rows[1]["per_class_f1_json"]
