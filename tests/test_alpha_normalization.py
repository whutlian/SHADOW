import torch

from shadow_hgc.demand.aggregate import aggregate_relation_demand, weighted_scatter_add
from shadow_hgc.demand.normalize import destination_row_normalize


def test_destination_row_alpha_sums_to_one_per_nonzero_destination():
    edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long)
    raw_weight = torch.tensor([1.0, 3.0, 2.0, 2.0])

    alpha = destination_row_normalize(edge_index, num_dst_nodes=3, raw_edge_weight=raw_weight)

    assert torch.allclose(alpha, torch.tensor([0.25, 0.75, 0.5, 0.5]))
    dst_sum = torch.zeros(3).index_add(0, edge_index[1], alpha)
    assert torch.allclose(dst_sum, torch.tensor([1.0, 1.0, 0.0]))


def test_alpha_is_not_source_degree_normalized():
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)

    alpha = destination_row_normalize(edge_index, num_dst_nodes=2)

    assert torch.allclose(alpha, torch.tensor([1.0, 0.5, 0.5]))


def test_weighted_scatter_add_matches_explicit_sum():
    messages = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    dst = torch.tensor([0, 0, 1], dtype=torch.long)
    weight = torch.tensor([0.25, 0.75, 1.0])

    out = weighted_scatter_add(messages, dst, weight, num_dst_nodes=2)

    expected = torch.tensor([[2.5, 1.5], [0.0, 4.0]])
    assert torch.allclose(out, expected)


def test_demand_aggregation_matches_hand_computed_mu():
    edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    source_phi = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    raw_weight = torch.tensor([1.0, 3.0, 2.0])

    demand, alpha = aggregate_relation_demand(
        edge_index=edge_index,
        source_features=source_phi,
        num_dst_nodes=2,
        raw_edge_weight=raw_weight,
        return_alpha=True,
    )

    assert torch.allclose(alpha, torch.tensor([0.25, 0.75, 1.0]))
    assert torch.allclose(demand, torch.tensor([[2.5, 0.0], [0.0, 4.0]]))


def test_chunked_demand_aggregation_matches_full_aggregation():
    edge_index = torch.tensor([[0, 1, 2, 0, 3], [0, 0, 1, 2, 2]], dtype=torch.long)
    source_phi = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
            [0.0, 4.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    raw_weight = torch.tensor([1.0, 3.0, 2.0, 1.0, 1.0])

    full, alpha = aggregate_relation_demand(
        edge_index=edge_index,
        source_features=source_phi,
        num_dst_nodes=3,
        raw_edge_weight=raw_weight,
        return_alpha=True,
    )
    chunked, chunked_alpha = aggregate_relation_demand(
        edge_index=edge_index,
        source_features=source_phi,
        num_dst_nodes=3,
        alpha=alpha,
        edge_chunk_size=2,
        return_alpha=True,
    )

    assert torch.allclose(chunked_alpha, alpha)
    assert torch.allclose(chunked, full)
