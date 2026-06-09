from __future__ import annotations

import torch

from shadow_hgc.anchors.source_anchors import select_source_anchors
from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved


def test_source_anchors_are_exposed_under_original_schema_type():
    relation = DirectedRelation("actor", "acts_in", "movie")
    result = select_source_anchors(
        edge_index_source_to_target=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        relation=relation,
        train_target_mask=torch.tensor([True, True]),
        train_labels=torch.tensor([0, 1]),
        num_source_nodes=2,
        num_classes=2,
        max_anchors=1,
    )

    assert result.exposed_source_type == "actor"
    assert result.exposed_relation == relation
    assert ensure_schema_preserved(
        exposed_node_types=["actor", "movie"],
        exposed_relations=[result.exposed_relation],
        original_node_types=["actor", "movie"],
        original_relations=[relation],
    )
