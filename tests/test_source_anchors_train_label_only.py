from __future__ import annotations

import torch

from shadow_hgc.anchors.source_anchors import select_source_anchors
from shadow_hgc.data.schemas import DirectedRelation


def test_source_anchor_scores_use_train_target_labels_only():
    relation = DirectedRelation("keyword", "tagged", "movie")
    edges = torch.tensor([[0, 0, 1, 1], [0, 2, 1, 3]], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])
    labels_a = torch.tensor([0, 1, 0, 1])
    labels_b = torch.tensor([0, 1, 1, 0])

    first = select_source_anchors(edges, relation, train_mask, labels_a, num_source_nodes=2, num_classes=2, max_anchors=2)
    second = select_source_anchors(edges, relation, train_mask, labels_b, num_source_nodes=2, num_classes=2, max_anchors=2)

    assert torch.equal(first.anchor_indices, second.anchor_indices)
    assert torch.allclose(first.scores, second.scores)
