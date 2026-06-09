from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.features.path_label_affinity import compute_path_label_affinity


class TinyGraph:
    def __init__(self, relation: DirectedRelation) -> None:
        self.edge_index = {relation: torch.tensor([[0], [0]], dtype=torch.long)}
        self.num_nodes = {"director": 1, "movie": 1}


def test_path_lad_returns_feature_block_not_metapath_edge_type():
    relation = DirectedRelation("director", "directed", "movie")
    block = compute_path_label_affinity(
        TinyGraph(relation),
        target_type="movie",
        path=[relation],
        train_target_mask=torch.tensor([True]),
        train_labels=torch.tensor([0]),
        num_classes=1,
    )

    assert block.shape == (1, 1)
    assert ensure_schema_preserved(
        exposed_node_types=["director", "movie"],
        exposed_relations=[relation],
        original_node_types=["director", "movie"],
        original_relations=[relation],
    )
