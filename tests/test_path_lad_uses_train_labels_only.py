from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.path_label_affinity import compute_path_label_affinity


class TinyGraph:
    def __init__(self, relation: DirectedRelation) -> None:
        self.edge_index = {
            relation: torch.tensor(
                [
                    [0, 0, 1, 1],
                    [0, 1, 1, 2],
                ],
                dtype=torch.long,
            )
        }
        self.num_nodes = {"actor": 2, "movie": 3}


def test_path_lad_uses_train_labels_only():
    relation = DirectedRelation("actor", "acts_in", "movie")
    graph = TinyGraph(relation)
    train_mask = torch.tensor([True, True, False])
    labels_a = torch.tensor([0, 1, 0])
    labels_b = torch.tensor([0, 1, 1])

    lad_a = compute_path_label_affinity(
        graph,
        target_type="movie",
        path=[relation],
        train_target_mask=train_mask,
        train_labels=labels_a,
        num_classes=2,
        leave_one_out_for_train=True,
    )
    lad_b = compute_path_label_affinity(
        graph,
        target_type="movie",
        path=[relation],
        train_target_mask=train_mask,
        train_labels=labels_b,
        num_classes=2,
        leave_one_out_for_train=True,
    )

    assert torch.allclose(lad_a, lad_b)
