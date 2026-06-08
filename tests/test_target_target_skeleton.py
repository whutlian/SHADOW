import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.skeleton.transition import compute_target_target_residual_skeleton, compute_transition_mass, topk_skeleton


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


def test_target_target_skeleton_uses_sparse_topk_without_dense_transition_snapshots():
    demand = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    prototype_features = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
        ]
    )
    target_to_cell = torch.arange(4, dtype=torch.long)
    cell_members = [torch.tensor([idx], dtype=torch.long) for idx in range(4)]
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [2, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    alpha = torch.tensor([0.2, 0.5, 0.3, 0.4, 0.6])

    result = compute_target_target_residual_skeleton(
        demand=demand,
        prototype_features=prototype_features,
        target_to_cell=target_to_cell,
        cell_members=cell_members,
        edge_index=edge_index,
        alpha=alpha,
        k_s=2,
    )

    assert result.S.numel() == 0
    assert result.S_top.numel() == 0
    assert torch.equal(result.skeleton_edge_index, torch.tensor([[1, 2, 0, 3], [2, 2, 3, 3]], dtype=torch.long))
    assert torch.allclose(result.skeleton_edge_weight, torch.tensor([0.5, 0.3, 0.6, 0.4]))
    expected_skel = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [9.0, 19.0],
            [22.0, 16.0],
        ]
    )
    assert torch.allclose(result.residual, demand - expected_skel)
    assert torch.isclose(torch.tensor(result.skeleton_mass_coverage), torch.tensor(0.9))
