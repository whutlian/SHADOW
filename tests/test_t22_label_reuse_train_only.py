from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank import compute_label_reuse_blocks


def test_label_reuse_ignores_valid_and_test_labels():
    rel = DirectedRelation("paper", "cite_ref", "paper")
    edge_index = torch.tensor([[0, 2, 3], [1, 1, 2]], dtype=torch.long)
    train = torch.tensor([0], dtype=torch.long)
    labels_a = torch.tensor([1, 0, 2, 2], dtype=torch.long)
    labels_b = torch.tensor([1, 2, 0, 0], dtype=torch.long)

    blocks_a, diag_a = compute_label_reuse_blocks(
        relation_blocks={"cite_ref": edge_index},
        labels=labels_a,
        train_target_ids=train,
        num_target_nodes=4,
        num_classes=3,
        steps=(1, 2),
        edge_chunk_size=2,
    )
    blocks_b, diag_b = compute_label_reuse_blocks(
        relation_blocks={"cite_ref": edge_index},
        labels=labels_b,
        train_target_ids=train,
        num_target_nodes=4,
        num_classes=3,
        steps=(1, 2),
        edge_chunk_size=2,
    )

    assert torch.allclose(blocks_a["Y1_cite_ref"], blocks_b["Y1_cite_ref"])
    assert torch.allclose(blocks_a["Y2_cite_ref"], blocks_b["Y2_cite_ref"])
    assert diag_a["uses_valid_labels"] is False
    assert diag_b["uses_test_labels"] is False


def test_label_reuse_prior_centering_uses_train_prior_only():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    labels = torch.tensor([1, 1, 0], dtype=torch.long)
    blocks, diag = compute_label_reuse_blocks(
        relation_blocks={"r": edge_index},
        labels=labels,
        train_target_ids=torch.tensor([0, 1], dtype=torch.long),
        num_target_nodes=3,
        num_classes=2,
        steps=(1,),
        prior_centering=True,
        edge_chunk_size=2,
    )

    assert torch.allclose(torch.tensor(diag["train_label_prior"]), torch.tensor([0.0, 1.0]))
    assert "Y1_r_centered" in blocks
    assert blocks["Y1_r_centered"].shape == (3, 2)
