from __future__ import annotations

import torch

from shadow_hgc.features.typed_feature_demand import compute_typed_feature_demand


def test_typed_feature_demand_matches_hand_dense_destination_row_alpha():
    edge_index = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)
    source_features = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    result = compute_typed_feature_demand(
        edge_index=edge_index,
        source_features=source_features,
        num_target_nodes=2,
        target_rows=torch.tensor([0, 1]),
        chunk_size=1,
    )

    expected = torch.tensor([[0.5, 1.0], [0.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(result.block, expected, atol=1e-6)
    assert result.diagnostics["normalization"] == "destination_row"
    assert result.diagnostics["full_edge_scans"] == 1
