from __future__ import annotations

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schema_audit import audit_dblp_schema
from shadow_hgc.data.schemas import DirectedRelation


def test_dblp_schema_audit_requires_author_target_and_apa_path():
    relation = DirectedRelation("paper", "written_by", "author")
    graph = HeteroGraphData(
        dataset_name="dblp",
        target_type="author",
        node_features={"paper": torch.zeros(3, 2), "author": torch.zeros(2, 2)},
        edge_index={relation: torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)},
        labels=torch.tensor([0, 1]),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([], dtype=torch.long),
        test_idx=torch.tensor([1]),
        relations=[relation],
        num_nodes={"paper": 3, "author": 2},
    )

    audit = audit_dblp_schema(graph, requested_metapath_blocks=["APA"])

    assert audit["target_type"] == "author"
    assert audit["apa_available"] is True
    assert audit["computed_metapath_blocks"] == ["APA"]
    assert audit["hard_requirements_passed"] is True
