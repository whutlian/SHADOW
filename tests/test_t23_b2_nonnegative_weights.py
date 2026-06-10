import torch

from shadow_hgc.condense.sft_condense import nonnegative_b2_weights


def test_t23_b2_weights_are_nonnegative():
    demand = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    shadows = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    indices, weights = nonnegative_b2_weights(demand, shadows)
    assert indices.shape == weights.shape == (2, 2)
    assert torch.all(weights >= 0)
