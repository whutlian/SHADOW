import torch

from shadow_hgc.shadows.assign import assign_nearest_shadow, build_b1_shadow_edges
from shadow_hgc.shadows.factorize import factorize_shadows


def test_residual_shadow_features_may_be_signed_but_edge_weights_are_nonnegative():
    residual = torch.tensor([[-1.0, 2.0], [3.0, -4.0], [0.5, -0.5]])

    shadows = factorize_shadows(residual, num_shadows=3, seed=7)
    assignment = assign_nearest_shadow(residual, shadows)
    edge_index, edge_weight = build_b1_shadow_edges(assignment)

    assert torch.any(shadows < 0.0)
    assert torch.all(edge_weight >= 0.0)
    assert torch.equal(edge_weight, torch.ones(3))
    assert torch.equal(edge_index[1], torch.arange(3))
