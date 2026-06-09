from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.metapath_table import compute_metapath_feature


def test_metapath_table_reports_no_dense_adjacency():
    rel = DirectedRelation("paper", "cite_ref", "paper")
    block, diagnostics = compute_metapath_feature(
        path_schema=[rel],
        target_type="paper",
        feature_provider={"paper": torch.eye(3)},
        edge_store={rel: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)},
        num_nodes={"paper": 3},
        target_rows=torch.tensor([0, 1, 2]),
        chunk_size=1,
    )

    assert block.shape == (3, 3)
    assert diagnostics["no_dense_adjacency"] is True
    assert diagnostics["materialized_dense_adjacency"] is False
