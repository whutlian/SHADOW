import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.shadow_fusion import ShadowFusionClassifier


def test_shadow_fusion_logs_positive_relation_and_block_gates():
    relation = DirectedRelation("author", "writes", "paper")
    model = ShadowFusionClassifier(
        in_channels={"author": 1, "paper": 1},
        out_channels=2,
        node_types=["author", "paper"],
        relations=[relation],
        target_type="paper",
        block_in_channels={"metapath:MAM": 1},
        relation_gate=True,
        block_gate=True,
        relation_gate_init=0.25,
        block_gate_init=0.5,
        hidden_dim=3,
        dropout=0.0,
    )

    diagnostics = model.diagnostics()

    assert diagnostics["final_logits_activation"] == "none"
    assert diagnostics["relation_gates"][str(relation)] > 0.0
    assert diagnostics["block_gates"]["metapath:MAM"] > 0.0


def test_shadow_fusion_relation_gate_scales_explicit_weighted_message():
    relation = DirectedRelation("author", "writes", "paper")
    model = ShadowFusionClassifier(
        in_channels={"author": 1, "paper": 1},
        out_channels=1,
        node_types=["author", "paper"],
        relations=[relation],
        target_type="paper",
        relation_gate=True,
        relation_gate_init=2.0,
        hidden_dim=0,
        dropout=0.0,
        bias=False,
    )
    with torch.no_grad():
        model.self_mlps["paper"].weight.zero_()
        model.relation_mlps[str(relation)].weight.fill_(1.0)

    logits = model(
        {"author": torch.tensor([[3.0], [5.0]]), "paper": torch.zeros(1, 1)},
        {relation: torch.tensor([[0, 1], [0, 0]], dtype=torch.long)},
        {relation: torch.tensor([0.25, 0.75])},
    )["paper"]
    gate = model.diagnostics()["relation_gates"][str(relation)]

    assert torch.allclose(logits, torch.tensor([[(3.0 * 0.25 + 5.0 * 0.75) * gate]]), atol=1e-6)
