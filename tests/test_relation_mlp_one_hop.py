import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.weighted_rel_linear import RelationMessageEncoderMLP


def test_relation_mlp_uses_one_hop_messages_without_second_message_passing():
    relation = DirectedRelation("paper", "cite_ref", "paper")
    model = RelationMessageEncoderMLP(
        in_channels={"paper": 2},
        out_channels=2,
        node_types=["paper"],
        relations=[relation],
        hidden_dim=4,
        dropout=0.0,
    )
    x = {"paper": torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])}
    edge_index = {relation: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}
    edge_weight = {relation: torch.ones(2)}

    encoded = model.encode_messages(x, edge_index, edge_weight)

    # Node 2 receives only node 1's original feature, not node 0 through node 1.
    relation_block = encoded["paper"][:, 2:]
    assert torch.allclose(relation_block[2], x["paper"][1])
    assert not torch.allclose(relation_block[2], x["paper"][0])


def test_relation_mlp_chunked_forward_matches_full_forward():
    relation = DirectedRelation("paper", "cite_ref", "paper")
    model = RelationMessageEncoderMLP(
        in_channels={"paper": 3},
        out_channels=2,
        node_types=["paper"],
        relations=[relation],
        hidden_dim=5,
        dropout=0.0,
    )
    model.eval()
    x = {"paper": torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10.0}
    edge_index = {relation: torch.tensor([[0, 1, 4, 5], [2, 2, 3, 3]], dtype=torch.long)}
    edge_weight = {relation: torch.tensor([0.25, 0.75, 0.4, 0.6], dtype=torch.float32)}

    full = model(x, edge_index, edge_weight)["paper"]
    chunked = model(x, edge_index, edge_weight, edge_chunk_size=2)["paper"]

    assert torch.allclose(chunked, full, atol=1e-6)
