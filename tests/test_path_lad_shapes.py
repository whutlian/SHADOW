from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.path_label_affinity import compute_path_label_affinity


class TinyGraph:
    def __init__(self, relation: DirectedRelation) -> None:
        self.edge_index = {relation: torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long)}
        self.num_nodes = {"keyword": 2, "movie": 4}


def test_path_lad_returns_requested_target_rows_by_num_classes():
    relation = DirectedRelation("keyword", "tagged", "movie")
    block = compute_path_label_affinity(
        TinyGraph(relation),
        target_type="movie",
        path=[relation],
        train_target_mask=torch.tensor([True, False, True, False]),
        train_labels=torch.tensor([0, -1, 2, -1]),
        num_classes=3,
        target_nodes=torch.tensor([2, 3]),
    )

    assert block.shape == (2, 3)
    assert torch.isfinite(block).all()
