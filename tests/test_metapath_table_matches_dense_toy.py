from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.metapath_table import compute_metapath_feature


def test_metapath_table_matches_hand_dense_target_source_target():
    paper_author = DirectedRelation("paper", "written_by", "author")
    author_paper = DirectedRelation("author", "writes", "paper")
    edge_store = {
        paper_author: torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
        author_paper: torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
    }
    features = {"paper": torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])}

    block, diagnostics = compute_metapath_feature(
        path_schema=[paper_author, author_paper],
        target_type="paper",
        feature_provider=features,
        edge_store=edge_store,
        num_nodes={"paper": 3, "author": 2},
        target_rows=torch.tensor([0, 1, 2]),
    )

    expected = torch.tensor([[2.0, 2.0], [0.0, 0.0], [0.5, 1.0]])
    assert torch.allclose(block, expected, atol=1e-6)
    assert diagnostics["materialized_dense_adjacency"] is False
