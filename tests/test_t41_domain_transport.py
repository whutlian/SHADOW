from __future__ import annotations

import numpy as np

from shadow_hgc.sft.domain_transport import (
    DomainTransportConfig,
    apply_domain_transport_to_selection,
    allocate_domain_row_budget,
    build_domain_bucket_stats,
    build_domain_transport_rows,
    compute_bucket_deficits,
    compute_domain_transport_strength,
)


def test_t41_domain_transport_activation_requires_gap_and_capacity() -> None:
    cfg = DomainTransportConfig()

    active = compute_domain_transport_strength(0.22, 96.0, cfg)
    low_gap = compute_domain_transport_strength(0.02, 96.0, cfg)
    low_capacity = compute_domain_transport_strength(0.22, 4.0, cfg)

    assert active >= 0.25
    assert low_gap < active
    assert low_gap < 0.10
    assert low_capacity < active
    assert low_capacity < 0.10


def test_t41_domain_row_budget_counts_inside_total_budget() -> None:
    cfg = DomainTransportConfig()
    rows, frac = allocate_domain_row_budget(
        total_budget=100,
        domain_transport_strength=1.0,
        scale_bucket="medium",
        cfg=cfg,
        min_core_rows=47,
    )

    assert rows == 30
    assert frac == cfg.max_domain_frac_medium

    capped, _ = allocate_domain_row_budget(
        total_budget=50,
        domain_transport_strength=1.0,
        scale_bucket="medium",
        cfg=cfg,
        min_core_rows=47,
    )

    assert capped == 3


def test_t41_bucket_stats_and_rows_ignore_valid_and_test_labels() -> None:
    cfg = DomainTransportConfig(projection_dim=2, num_quantile_bins=4, seed=7)
    signatures = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        dtype=np.float32,
    )
    train_idx = np.asarray([0, 1, 2], dtype=np.int64)
    train_labels = np.asarray([0, 1, 0], dtype=np.int64)
    target_idx = np.arange(signatures.shape[0], dtype=np.int64)

    stats_a = build_domain_bucket_stats(
        signatures,
        target_idx=target_idx,
        train_idx=train_idx,
        train_labels=train_labels,
        cfg=cfg,
        valid_labels=np.asarray([9, 9]),
        test_labels=np.asarray([9, 9]),
    )
    stats_b = build_domain_bucket_stats(
        signatures,
        target_idx=target_idx,
        train_idx=train_idx,
        train_labels=train_labels,
        cfg=cfg,
        valid_labels=np.asarray([1, 2]),
        test_labels=np.asarray([3, 4]),
    )

    assert np.array_equal(stats_a.bucket_ids, stats_b.bucket_ids)
    assert stats_a.all_bucket_mass == stats_b.all_bucket_mass
    assert stats_a.train_bucket_mass == stats_b.train_bucket_mass

    rows_a = build_domain_transport_rows(
        bucket_ids=stats_a.bucket_ids,
        train_rows=train_idx,
        train_labels=train_labels,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        total_budget=4,
        num_classes=2,
        domain_transport_strength=1.0,
        cfg=cfg,
    )
    rows_b = build_domain_transport_rows(
        bucket_ids=stats_b.bucket_ids,
        train_rows=train_idx,
        train_labels=train_labels,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        total_budget=4,
        num_classes=2,
        domain_transport_strength=1.0,
        cfg=cfg,
    )

    assert np.array_equal(rows_a.anchor_rows, rows_b.anchor_rows)
    assert rows_a.row_type_counts["domain_transport"] == rows_a.domain_transport_rows


def test_t41_compute_deficits_and_apply_selection_keep_ratio_accounting() -> None:
    bucket_ids = np.asarray([1, 1, 2, 2, 3, 3, 3, 4], dtype=np.uint64)
    train_rows = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    train_labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    selected = np.asarray([0, 1, 2, 3], dtype=np.int64)

    deficits = compute_bucket_deficits({1: 0.50}, {1: 0.25, 2: 0.25, 3: 0.375, 4: 0.125})
    assert deficits[3] > deficits[2] > 0.0

    out = apply_domain_transport_to_selection(
        base_selected_rows=selected,
        bucket_ids=bucket_ids,
        train_rows=train_rows,
        train_labels=train_labels,
        total_budget=6,
        num_classes=2,
        domain_gap_train_all=0.30,
        cfg=DomainTransportConfig(seed=9, capacity_threshold=2.0),
    )

    assert out.selected_rows.shape[0] == 6
    assert out.row_type_counts["domain_transport"] == out.domain_transport_rows
    assert out.row_type_counts["hard_anchor"] + out.row_type_counts["domain_transport"] == 6
    assert out.domain_gap_after <= out.domain_gap_before


def test_t41_domain_transport_module_has_no_all_pair_dense_path() -> None:
    import inspect
    import shadow_hgc.sft.domain_transport as domain_transport

    source = inspect.getsource(domain_transport)

    assert "torch.cdist" not in source
    assert "cdist(" not in source
    assert "uses_dense_all_node_teacher_cache" not in source
