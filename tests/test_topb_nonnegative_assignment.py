import torch

from shadow_hgc.shadows.assign import topb_nonnegative_assignment


def test_topb_assignment_produces_nonnegative_edge_weights_and_reconstructs_rows():
    demand = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    shadows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])

    result = topb_nonnegative_assignment(demand, shadows, b=2, ridge_lambda=1e-4)

    assert result.edge_index.shape[0] == 2
    assert torch.all(result.edge_weight >= 0.0)
    assert result.reconstruction.shape == demand.shape
    assert torch.linalg.norm(demand - result.reconstruction) < torch.linalg.norm(demand)


def test_topb_assignment_allows_signed_shadow_features_but_not_negative_weights():
    demand = torch.tensor([[-1.0, 0.5]])
    shadows = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])

    result = topb_nonnegative_assignment(demand, shadows, b=2)

    assert torch.any(shadows < 0.0)
    assert torch.all(result.edge_weight >= 0.0)
