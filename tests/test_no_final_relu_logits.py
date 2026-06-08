import pytest
import torch
import torch.nn.functional as F

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.factory import build_model, final_logits_activation_status
from shadow_hgc.models.weighted_rel_linear import WeightedRelationLinearConv


def test_factory_relation_linear_logits_have_no_final_relu_and_can_be_negative():
    model, diagnostics = build_model(
        model_type="relation_linear",
        in_channels={"paper": 2},
        out_channels=2,
        node_types=["paper"],
        relations=[],
    )
    with torch.no_grad():
        model.self_linears["paper"].weight.copy_(torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
        model.self_linears["paper"].bias.zero_()

    logits = model({"paper": torch.tensor([[2.0, -0.5]])}, {}, {})["paper"]
    loss = F.cross_entropy(logits, torch.tensor([1]))

    assert diagnostics["final_logits_activation"] == "none"
    assert logits.min().item() < 0.0
    assert torch.allclose(loss, F.cross_entropy(torch.tensor([[-2.0, -0.5]]), torch.tensor([1])))


def test_unsafe_final_relu_logits_are_rejected_by_diagnostics():
    relation = DirectedRelation("paper", "cite_ref", "paper")
    unsafe = WeightedRelationLinearConv(
        in_channels={"paper": 2},
        out_channels=3,
        node_types=["paper"],
        relations=[relation],
    )

    assert final_logits_activation_status(unsafe, used_as_logits=True) == "unsafe_relu_logits"
    with pytest.raises(ValueError, match="unsafe final ReLU"):
        build_model(
            model_type="relation_linear",
            in_channels={"paper": 2},
            out_channels=3,
            node_types=["paper"],
            relations=[relation],
            final_activation="relu",
        )


def test_factory_relation_mlp_and_shadow_fusion_report_raw_final_logits():
    relation = DirectedRelation("author", "writes", "paper")

    for model_type in ("relation_mlp", "shadow_fusion"):
        model, diagnostics = build_model(
            model_type=model_type,
            in_channels={"author": 2, "paper": 2},
            out_channels=2,
            node_types=["author", "paper"],
            relations=[relation],
            target_type="paper",
            hidden_dim=4,
            dropout=0.0,
        )

        assert diagnostics["final_logits_activation"] == "none"
        last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
        assert isinstance(last_linear, torch.nn.Linear)
