from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.path_label_affinity import compute_path_label_affinity


class TinyGraph:
    def __init__(self, relation: DirectedRelation) -> None:
        self.edge_index = {relation: torch.tensor([[0, 0], [0, 1]], dtype=torch.long)}
        self.num_nodes = {"actor": 1, "movie": 2}


def test_path_lad_leave_one_out_removes_target_train_label_from_cycle():
    relation = DirectedRelation("actor", "acts_in", "movie")
    graph = TinyGraph(relation)
    train_mask = torch.tensor([True, True])
    labels = torch.tensor([0, 1])

    lad = compute_path_label_affinity(
        graph,
        target_type="movie",
        path=[relation],
        train_target_mask=train_mask,
        train_labels=labels,
        num_classes=2,
        target_nodes=torch.tensor([0]),
        leave_one_out_for_train=True,
        normalize="none",
    )

    assert lad.shape == (1, 2)
    assert lad[0, 0].item() == 0.0
    assert lad[0, 1].item() > 0.0
