from __future__ import annotations

import torch

from shadow_hgc.features.label_affinity import (
    aggregate_non_target_label_affinity,
    compute_source_label_counts,
    compute_target_target_label_affinity,
    normalize_label_affinity_block,
)


def test_lad_and_normalization_outputs_have_no_nan_for_zero_rows():
    train_target_mask = torch.tensor([False, False, False])
    train_labels = torch.tensor([10, 11, 12], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    target_block = compute_target_target_label_affinity(
        edge_index,
        train_target_mask,
        train_labels,
        num_nodes=3,
        num_classes=2,
    )
    source_affinity = compute_source_label_counts(
        edge_index,
        train_target_mask,
        train_labels,
        num_source_nodes=2,
        num_classes=2,
    )
    non_target_block = aggregate_non_target_label_affinity(
        edge_index,
        source_affinity,
        target_nodes=torch.arange(3),
    )

    assert torch.isfinite(target_block).all()
    assert torch.isfinite(source_affinity.counts).all()
    assert torch.isfinite(non_target_block).all()

    block = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    for mode in ("none", "row_l1", "standardize", "standardize_l2"):
        normalized, stats = normalize_label_affinity_block(block, mode=mode)
        assert torch.isfinite(normalized).all()
        assert torch.isfinite(stats.mean).all()
        assert torch.isfinite(stats.std).all()
        assert stats.l1_norm_mean >= 0.0
        assert stats.l2_norm_mean >= 0.0
        assert stats.zero_row_ratio == 2.0 / 3.0
