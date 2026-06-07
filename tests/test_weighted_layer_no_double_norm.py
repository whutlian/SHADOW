import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.weighted_rel_linear import WeightedRelationLinearConv


def test_weighted_relation_linear_equals_explicit_weighted_scatter_add():
    relation = DirectedRelation("author", "writes", "paper")
    conv = WeightedRelationLinearConv(
        in_channels={"author": 2, "paper": 2},
        out_channels=2,
        node_types=["author", "paper"],
        relations=[relation],
        activation=None,
        bias=False,
    )
    with torch.no_grad():
        conv.self_linears["paper"].weight.zero_()
        conv.self_linears["author"].weight.zero_()
        conv.relation_linears[str(relation)].weight.copy_(torch.eye(2))

    x_dict = {
        "author": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "paper": torch.zeros(2, 2),
    }
    edge_index = {relation: torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)}
    edge_weight = {relation: torch.tensor([0.25, 0.75, 1.0])}

    out = conv(x_dict, edge_index, edge_weight)

    expected_paper = torch.tensor([[2.5, 3.5], [3.0, 4.0]])
    assert torch.allclose(out["paper"], expected_paper)


def test_weighted_relation_linear_chunked_edges_equal_full_scatter():
    relation = DirectedRelation("paper", "cite_ref", "paper")
    conv = WeightedRelationLinearConv(
        in_channels={"paper": 3},
        out_channels=2,
        node_types=["paper"],
        relations=[relation],
        activation=None,
        bias=False,
    )
    with torch.no_grad():
        conv.self_linears["paper"].weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        conv.relation_linears[str(relation)].weight.copy_(
            torch.tensor([[0.5, 0.0, 1.0], [0.0, -1.0, 0.25]])
        )

    x_dict = {
        "paper": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0],
                [4.0, 0.0, 1.0],
            ]
        )
    }
    edge_index = {relation: torch.tensor([[0, 1, 2, 0, 2], [1, 1, 0, 2, 2]], dtype=torch.long)}
    edge_weight = {relation: torch.tensor([0.25, 0.75, 1.0, 0.5, 0.5])}

    full = conv(x_dict, edge_index, edge_weight)
    chunked = conv(x_dict, edge_index, edge_weight, edge_chunk_size=2)

    assert torch.allclose(chunked["paper"], full["paper"])
