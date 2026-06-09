from __future__ import annotations

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.pipeline.core import _build_path_lad_blocks


def test_core_path_lad_builds_p2_for_target_target_relation():
    relation = DirectedRelation("paper", "cited_by", "paper")
    graph = HeteroGraphData(
        dataset_name="ogbn-arxiv",
        target_type="paper",
        node_features={"paper": torch.zeros(4, 2)},
        edge_index={relation: torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)},
        labels=torch.tensor([0, 1, -1, -1]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([], dtype=torch.long),
        test_idx=torch.tensor([2, 3]),
        relations=[relation],
        num_nodes={"paper": 4},
    )

    blocks, stats, meta = _build_path_lad_blocks(
        graph,
        [relation],
        target_nodes=torch.arange(4),
        num_classes=2,
        requested_blocks=["P1", "P2"],
        normalize="row_l1",
        leave_one_out_for_train=True,
    )

    assert sorted(blocks) == ["P1", "P2"]
    assert stats["P2"]["path_length"] == 2
    assert meta["path_lad_blocks"] == ["P1", "P2"]
