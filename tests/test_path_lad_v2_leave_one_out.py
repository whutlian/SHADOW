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


def test_path_lad_v2_leave_one_out_removes_train_row_own_label():
    relation = DirectedRelation("actor", "acts_in", "movie")
    result = compute_path_lad_v2_blocks(
        TinyGraph(relation),
        requested_paths=["MAM"],
        train_target_mask=torch.tensor([True, True, False]),
        train_labels=torch.tensor([0, 1, -1]),
        num_classes=2,
        leave_one_out=True,
    )

    block = result.blocks["MAM"]
    assert block[0, 0].item() == 0.0
    assert result.diagnostics["path_lad_leave_one_out"] is True
