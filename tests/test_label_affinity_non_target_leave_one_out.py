from __future__ import annotations

import torch

from shadow_hgc.features.label_affinity import aggregate_non_target_label_affinity, compute_source_label_counts


def test_non_target_lad_leave_one_out_for_train_targets_only():
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2],
            [0, 1, 0, 3, 2],
        ],
        dtype=torch.long,
    )
    train_target_mask = torch.tensor([True, True, False, False])
    train_labels = torch.tensor([0, 1, 99, 88], dtype=torch.long)
    source_affinity = compute_source_label_counts(
        source_to_target_edges=edge_index,
        train_target_mask=train_target_mask,
        train_labels=train_labels,
        num_source_nodes=3,
        num_classes=2,
    )

    lad = aggregate_non_target_label_affinity(
        edge_index_source_to_target=edge_index,
        source_affinity=source_affinity,
        target_nodes=torch.tensor([0, 2, 3], dtype=torch.long),
        target_train_labels=train_labels,
        leave_one_out_for_train=True,
    )

    expected = torch.tensor(
        [
            [0.0, 0.5],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    assert torch.allclose(lad, expected)
