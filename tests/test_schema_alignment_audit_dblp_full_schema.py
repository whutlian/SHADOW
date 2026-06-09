from __future__ import annotations

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schema_audit import audit_schema_alignment
from shadow_hgc.data.schemas import DirectedRelation


def test_dblp_full_schema_audit_reports_required_edges_and_missing_paths():
    pa = DirectedRelation("paper", "written_by", "author")
    ap = DirectedRelation("author", "writes_rev", "paper")
    pv = DirectedRelation("paper", "published_in", "venue")
    vp = DirectedRelation("venue", "publishes_rev", "paper")
    graph = HeteroGraphData(
        dataset_name="dblp",
        target_type="author",
        node_features={"author": torch.zeros(2, 2), "paper": torch.zeros(3, 2), "venue": torch.zeros(1, 2)},
        edge_index={
            pa: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            ap: torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            pv: torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
            vp: torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        },
        labels=torch.tensor([0, 1]),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([], dtype=torch.long),
        test_idx=torch.tensor([1]),
        relations=[pa, ap, pv, vp],
        num_nodes={"author": 2, "paper": 3, "venue": 1},
    )

    row = audit_schema_alignment(graph, loader_name="full_schema", source="unit")

    assert row["target_type"] == "author"
    assert "APA" in row["metapath_available"]
    assert "APVPA" in row["metapath_available"]
    assert "APTPA" in row["metapath_missing"]
    assert row["freehgc_or_hgb_alignment_status"] == "partial"
