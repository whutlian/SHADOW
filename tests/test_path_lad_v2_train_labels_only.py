from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.path_lad_v2 import compute_path_lad_v2_blocks


class TinyGraph:
    def __init__(self, relation: DirectedRelation) -> None:
        self.dataset_name = "imdb"
        self.target_type = "movie"
        self.relations = [relation]
        self.edge_index = {relation: torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long)}
        self.num_nodes = {"actor": 2, "movie": 3}


def test_path_lad_v2_uses_train_labels_only():
    relation = DirectedRelation("actor", "acts_in", "movie")
    train_mask = torch.tensor([True, True, False])
    labels_a = torch.tensor([0, 1, 0])
    labels_b = torch.tensor([0, 1, 1])

    result_a = compute_path_lad_v2_blocks(
        TinyGraph(relation),
        requested_paths=["MAM"],
        train_target_mask=train_mask,
        train_labels=labels_a,
        num_classes=2,
    )
    result_b = compute_path_lad_v2_blocks(
        TinyGraph(relation),
        requested_paths=["MAM"],
        train_target_mask=train_mask,
        train_labels=labels_b,
        num_classes=2,
    )

    assert torch.allclose(result_a.blocks["MAM"], result_b.blocks["MAM"])
    assert result_a.diagnostics["path_lad_uses_val_or_test_labels"] is False
    assert result_a.diagnostics["path_lad_num_train_labels_used"] == 2
