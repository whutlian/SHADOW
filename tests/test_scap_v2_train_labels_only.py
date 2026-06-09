from __future__ import annotations

import torch

from shadow_hgc.features.scap_v2 import compute_target_target_scap_v2


def test_scap_v2_uses_train_labels_only_for_target_target_affinity():
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, -1], dtype=torch.long)
    train_mask = torch.tensor([True, False, False, False])

    block = compute_target_target_scap_v2(
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        num_nodes=4,
        num_classes=3,
        target_rows=torch.tensor([3]),
        prior_center=False,
    )

    assert block.dense is not None
    assert block.dense.tolist() == [[1.0 / 3.0, 0.0, 0.0]]
    assert block.diagnostics["uses_validation_labels"] is False
    assert block.diagnostics["uses_test_labels"] is False
