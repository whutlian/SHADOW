import torch

from shadow_hgc.logits.ensemble import nonnegative_grid_weights, weighted_logit_ensemble


def test_safe_logit_ensemble_weights_are_nonnegative_and_sum_to_one():
    weights = nonnegative_grid_weights(num_models=2, step=0.5)

    assert weights == [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    for row in weights:
        assert min(row) >= 0.0
        assert abs(sum(row) - 1.0) < 1e-9

    logits = weighted_logit_ensemble([torch.ones(2, 2), torch.zeros(2, 2)], weights[1])
    assert torch.allclose(logits, torch.full((2, 2), 0.5))
