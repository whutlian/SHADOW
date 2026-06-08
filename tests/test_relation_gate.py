import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.weighted_rel_linear import WeightedRelationLinearConv


def test_relation_gate_is_positive_after_softplus_and_scales_messages():
    relation = DirectedRelation("author", "writes", "paper")
    conv = WeightedRelationLinearConv(
        in_channels={"author": 1, "paper": 1},
        out_channels=1,
        node_types=["author", "paper"],
        relations=[relation],
        activation=None,
        bias=False,
        relation_gate=True,
        relation_gate_init=1.0,
    )
    with torch.no_grad():
        conv.self_linears["paper"].weight.zero_()
        conv.self_linears["author"].weight.zero_()
        conv.relation_linears[str(relation)].weight.fill_(1.0)

    gate = conv.relation_gate_values()[str(relation)]
    out = conv(
        {"author": torch.tensor([[2.0]]), "paper": torch.zeros(1, 1)},
        {relation: torch.tensor([[0], [0]], dtype=torch.long)},
        {relation: torch.tensor([1.0])},
    )

    assert gate > 0.0
    assert torch.allclose(out["paper"], torch.tensor([[2.0 * gate]]), atol=1e-6)
