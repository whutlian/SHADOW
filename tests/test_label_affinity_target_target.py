from __future__ import annotations

import torch

from shadow_hgc.features.label_affinity import compute_target_target_label_affinity


def test_target_target_lad_uses_destination_alpha_train_source_labels_and_drops_self_loops():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 2],
            [2, 2, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    train_target_mask = torch.tensor([True, False, True, True, False])
    train_labels = torch.tensor([0, 99, 1, 1, 42], dtype=torch.long)

    lad = compute_target_target_label_affinity(
        edge_index=edge_index,
        train_target_mask=train_target_mask,
        train_labels=train_labels,
        num_nodes=5,
        num_classes=2,
        exclude_self=True,
        target_nodes=torch.tensor([2, 3], dtype=torch.long),
    )

    expected = torch.tensor(
        [
            [1.0 / 4.0, 1.0 / 4.0],
            [0.0, 0.5],
        ]
    )
    assert torch.allclose(lad, expected)
