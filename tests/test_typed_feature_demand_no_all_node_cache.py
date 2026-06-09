from __future__ import annotations

import pytest
import torch

from shadow_hgc.features.typed_feature_demand import compute_typed_feature_demand


def test_typed_feature_demand_rejects_all_node_cache_without_debug():
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    source_features = torch.ones(1, 4)

    with pytest.raises(ValueError, match="all-node high-dimensional demand cache"):
        compute_typed_feature_demand(
            edge_index=edge_index,
            source_features=source_features,
            num_target_nodes=1,
            target_rows=torch.tensor([0]),
            cache_all_nodes=True,
            debug_allow_all_node_cache=False,
        )
