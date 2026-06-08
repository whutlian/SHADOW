from __future__ import annotations

import torch

from shadow_hgc.features.label_affinity import (
    aggregate_non_target_label_affinity,
    compute_source_label_counts,
    compute_target_target_label_affinity,
)


def test_lad_outputs_are_unchanged_when_validation_and_test_labels_are_permuted():
    train_target_mask = torch.tensor([True, True, False, False])
    labels_a = torch.tensor([0, 1, 99, 88], dtype=torch.long)
    labels_b = torch.tensor([0, 1, 77, 66], dtype=torch.long)

    target_target_edges = torch.tensor(
        [
            [0, 2, 1, 3],
            [2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    tt_a = compute_target_target_label_affinity(
        target_target_edges,
        train_target_mask,
        labels_a,
        num_nodes=4,
        num_classes=2,
        target_nodes=torch.tensor([2, 3], dtype=torch.long),
    )
    tt_b = compute_target_target_label_affinity(
        target_target_edges,
        train_target_mask,
        labels_b,
        num_nodes=4,
        num_classes=2,
        target_nodes=torch.tensor([2, 3], dtype=torch.long),
    )

    non_target_edges = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 2, 1, 3],
        ],
        dtype=torch.long,
    )
    source_a = compute_source_label_counts(
        non_target_edges,
        train_target_mask,
        labels_a,
        num_source_nodes=2,
        num_classes=2,
    )
    source_b = compute_source_label_counts(
        non_target_edges,
        train_target_mask,
        labels_b,
        num_source_nodes=2,
        num_classes=2,
    )
    nt_a = aggregate_non_target_label_affinity(
        non_target_edges,
        source_a,
        target_nodes=torch.tensor([2, 3], dtype=torch.long),
        target_train_labels=labels_a,
    )
    nt_b = aggregate_non_target_label_affinity(
        non_target_edges,
        source_b,
        target_nodes=torch.tensor([2, 3], dtype=torch.long),
        target_train_labels=labels_b,
    )

    assert torch.allclose(tt_a, tt_b)
    assert torch.allclose(nt_a, nt_b)
