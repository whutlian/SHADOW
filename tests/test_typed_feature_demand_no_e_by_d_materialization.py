from __future__ import annotations

import torch

from shadow_hgc.features.typed_feature_demand import compute_typed_feature_demand


def test_typed_feature_demand_does_not_materialize_full_e_by_d():
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    source_features = torch.randn(4, 8)

    result = compute_typed_feature_demand(
        edge_index=edge_index,
        source_features=source_features,
        num_target_nodes=2,
        target_rows=torch.tensor([0, 1]),
        chunk_size=2,
    )

    assert result.block.shape == (2, 8)
    assert result.diagnostics["materialized_full_e_by_d"] is False
    assert result.diagnostics["max_edge_chunk_size"] <= 2
