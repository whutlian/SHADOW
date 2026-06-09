from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.two_hop_lad import compute_two_hop_lad


def test_two_hop_lad_uses_train_labels_only_and_no_feature_diffusion():
    relation = DirectedRelation("paper", "cited_by", "paper")
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])
    labels_a = torch.tensor([0, 1, 0, 1])
    labels_b = torch.tensor([0, 1, 1, 0])

    first = compute_two_hop_lad(
        edge_index,
        num_nodes=4,
        train_target_mask=train_mask,
        train_labels=labels_a,
        num_classes=2,
        steps=2,
    )
    second = compute_two_hop_lad(
        edge_index,
        num_nodes=4,
        train_target_mask=train_mask,
        train_labels=labels_b,
        num_classes=2,
        steps=2,
    )

    assert torch.allclose(first.blocks["P1"], second.blocks["P1"])
    assert torch.allclose(first.blocks["P2"], second.blocks["P2"])
    assert first.diagnostics["uses_train_labels_only"] is True
    assert first.diagnostics["uses_feature_diffusion"] is False
