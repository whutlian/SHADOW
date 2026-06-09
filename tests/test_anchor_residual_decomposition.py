from __future__ import annotations

import torch

from shadow_hgc.anchors.source_anchors import anchor_residual_decomposition


def test_anchor_residual_decomposition_reconstructs_demand():
    demand = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    anchor_message = torch.tensor([[0.25, 1.5], [1.0, 2.0]])

    result = anchor_residual_decomposition(demand, anchor_message)

    assert torch.allclose(result.anchor_message + result.residual, demand)
    assert result.residual_energy_after <= result.residual_energy_before
