import torch

from shadow_hgc.shadows.assign import (
    assign_nearest_shadow,
    assign_nearest_shadow_chunked,
    topb_nonnegative_assignment,
    topb_nonnegative_assignment_chunked,
)


def test_chunked_b1_assignment_matches_dense_small_tensor():
    torch.manual_seed(42)
    demand = torch.randn(17, 5)
    shadows = torch.randn(9, 5)

    dense = assign_nearest_shadow(demand, shadows)
    chunked = assign_nearest_shadow_chunked(demand, shadows, chunk_size=4)

    assert torch.equal(chunked, dense)


def test_chunked_topb_nonnegative_assignment_matches_dense_small_tensor():
    torch.manual_seed(42)
    demand = torch.randn(13, 4)
    shadows = torch.randn(7, 4)

    dense = topb_nonnegative_assignment(demand, shadows, b=3, ridge_lambda=1e-4)
    chunked = topb_nonnegative_assignment_chunked(
        demand,
        shadows,
        b=3,
        ridge_lambda=1e-4,
        chunk_size=5,
    )

    assert torch.equal(chunked.topk_index, dense.topk_index)
    assert torch.allclose(chunked.topk_weight, dense.topk_weight, atol=1e-6)
    assert torch.allclose(chunked.reconstruction, dense.reconstruction, atol=1e-6)
    assert torch.equal(chunked.edge_index, dense.edge_index)
    assert torch.allclose(chunked.edge_weight, dense.edge_weight, atol=1e-6)
    assert torch.all(chunked.edge_weight >= 0.0)


def test_chunked_assignment_rejects_empty_shadow_pool():
    demand = torch.randn(3, 2)
    shadows = torch.empty(0, 2)

    try:
        assign_nearest_shadow_chunked(demand, shadows, chunk_size=2)
    except ValueError as exc:
        assert "empty shadow pool" in str(exc)
    else:
        raise AssertionError("expected empty shadow pool to fail")
