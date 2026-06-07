from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation


@dataclass
class HeteroGraphData:
    dataset_name: str
    target_type: str
    node_features: dict[str, torch.Tensor]
    edge_index: dict[DirectedRelation, torch.Tensor]
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    relations: list[DirectedRelation]
    num_nodes: dict[str, int]


def build_toy_graph(seed: int = 0) -> HeteroGraphData:
    torch.manual_seed(seed)
    target_type = "paper"
    cite_ref = DirectedRelation("paper", "cite_ref", "paper")
    cited_by = DirectedRelation("paper", "cited_by", "paper")
    writes = DirectedRelation("author", "writes", "paper")
    paper_x = torch.tensor(
        [
            [2.0, 0.0, 0.3, 1.0],
            [1.8, 0.2, 0.1, 0.9],
            [2.2, -0.2, 0.2, 1.1],
            [-1.8, 1.0, 0.5, -0.5],
            [-2.0, 1.2, 0.4, -0.7],
            [1.9, 0.1, 0.2, 0.8],
            [-1.7, 1.1, 0.6, -0.6],
            [0.2, -1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Direction is message_source -> message_destination.
    cite_edges = torch.tensor(
        [
            [1, 2, 5, 6, 0, 2, 7, 3, 5],
            [0, 0, 1, 3, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    cited_by_edges = torch.stack([cite_edges[1], cite_edges[0]], dim=0)
    writes_edges = torch.tensor(
        [
            [0, 0, 1, 1, 2, 3, 3, 4],
            [0, 1, 2, 5, 3, 4, 6, 7],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 0, 1, -1], dtype=torch.long)
    return HeteroGraphData(
        dataset_name="toy",
        target_type=target_type,
        node_features={"paper": paper_x},
        edge_index={cite_ref: cite_edges, cited_by: cited_by_edges, writes: writes_edges},
        labels=labels,
        train_idx=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        val_idx=torch.tensor([5], dtype=torch.long),
        test_idx=torch.tensor([6], dtype=torch.long),
        relations=[cite_ref, cited_by, writes],
        num_nodes={"paper": 8, "author": 5},
    )
