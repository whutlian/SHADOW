from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_hgc.ultra.papers100m_disco_parity import ensure_disco_baseline_csv, load_disco_baseline
from shadow_hgc.ultra.papers100m_t36_contract import (
    DISCO_PARITY_RATIOS,
    make_t36_row,
    percent_to_ratio_decimal,
    ratio_decimal_to_percent,
    validate_t36_row,
)


def test_t36_ratio_conversion_does_not_confuse_percent_and_decimal():
    assert percent_to_ratio_decimal(0.005) == 0.00005
    assert percent_to_ratio_decimal(0.010) == 0.00010
    assert percent_to_ratio_decimal(0.020) == 0.00020
    assert percent_to_ratio_decimal(0.050) == 0.00050
    assert ratio_decimal_to_percent(0.00005) == 0.005


def test_t36_disco_baseline_csv_has_all_fraction_rows(tmp_path: Path):
    path = tmp_path / "disco_papers100m_sgc.csv"
    ensure_disco_baseline_csv(path)
    rows = load_disco_baseline(path)

    assert set(rows) == set(DISCO_PARITY_RATIOS)
    for row in rows.values():
        for key in ("random_acc", "herding_acc", "kcenter_acc", "disco_acc", "whole_dataset_acc"):
            assert 0.0 <= float(row[key]) <= 1.0


def test_t36_promoted_row_rejects_forbidden_paths():
    row = make_t36_row(
        promotion_status="promoted",
        backend="sgc",
        requested_full_node_ratio=0.00005,
        uses_dense_all_node_teacher_cache=True,
        uses_exact_all_pair_distance=True,
    )

    result = validate_t36_row(row)

    assert result["valid"] is False
    assert "uses_dense_all_node_teacher_cache" in result["forbidden_flags"]
    assert "uses_exact_all_pair_distance" in result["forbidden_flags"]
