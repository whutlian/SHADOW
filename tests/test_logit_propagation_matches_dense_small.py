from __future__ import annotations

import torch

from shadow_hgc.features.logit_propagation import propagate_logits


def test_logit_propagation_matches_dense_destination_row_alpha():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]], dtype=torch.long)
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

    result = propagate_logits(
        edge_index=edge_index,
        logits=logits,
        num_nodes=3,
        target_rows=torch.tensor([0, 1, 2]),
        steps=1,
        lam=1.0,
        input_mode="logits",
    )

    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    assert torch.allclose(result.block, expected, atol=1e-6)
    assert result.diagnostics["propagates_features"] is False
