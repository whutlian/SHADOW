from __future__ import annotations

import torch

from shadow_hgc.features.scap import target_target_scap_dense


def test_scap_ignores_validation_and_test_labels():
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, -1], dtype=torch.long)
    train_mask = torch.tensor([True, False, False, False])

    block = target_target_scap_dense(
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        num_nodes=4,
        num_classes=3,
        target_rows=torch.tensor([3]),
    )

    assert block.tolist() == [[1.0 / 3.0, 0.0, 0.0]]
