import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.skeleton.transition import compute_transition_mass, topk_skeleton


def test_transition_mass_uses_the_same_alpha_scale_as_demand():
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long)
    alpha = destination_row_normalize(edge_index, num_dst_nodes=4)
    target_to_cell = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    cell_sizes = torch.tensor([2.0, 2.0])

    S = compute_transition_mass(
        edge_index=edge_index,
        alpha=alpha,
        target_to_cell=target_to_cell,
        cell_sizes=cell_sizes,
        num_cells=2,
    )

    assert torch.allclose(alpha, torch.tensor([0.5, 0.5, 1.0]))
    assert torch.allclose(S, torch.tensor([[0.5, 0.5], [0.0, 0.0]]))


def test_topk_skeleton_weights_are_not_renormalized_after_truncation():
    S = torch.tensor([[0.25, 0.50, 0.10], [0.20, 0.05, 0.00]])

    edge_index, edge_weight, S_top = topk_skeleton(S, k_s=1)

    assert torch.allclose(S_top, torch.tensor([[0.00, 0.50, 0.00], [0.20, 0.00, 0.00]]))
    assert torch.all(edge_weight <= 0.50)
    assert torch.allclose(edge_weight.sort().values, torch.tensor([0.20, 0.50]))
    assert torch.equal(edge_index, torch.tensor([[1, 0], [0, 1]], dtype=torch.long))
