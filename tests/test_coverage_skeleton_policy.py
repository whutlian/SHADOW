import torch

from shadow_hgc.skeleton.policy import coverage_topk_skeleton


def test_coverage_skeleton_does_not_renormalize_retained_mass():
    S = torch.tensor([[0.50, 0.25, 0.25], [0.10, 0.20, 0.00]])

    result = coverage_topk_skeleton(S, coverage=0.65, k_max=3)

    assert torch.allclose(result.S_top.sum(dim=1), torch.tensor([0.75, 0.20]))
    assert torch.allclose(result.edge_weight.sort().values, torch.tensor([0.20, 0.25, 0.50]))
    assert result.k_by_row == [2, 1]
    assert result.max_k == 2
    assert result.mean_k == 1.5
