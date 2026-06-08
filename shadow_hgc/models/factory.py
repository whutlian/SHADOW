from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.shadow_fusion import ShadowFusionClassifier
from shadow_hgc.models.weighted_rel_linear import RelationMessageEncoderMLP, WeightedRelationLinearConv


def final_logits_activation_status(model: nn.Module, *, used_as_logits: bool = True) -> str:
    if isinstance(model, WeightedRelationLinearConv):
        if model.activation is None:
            return "none"
        if used_as_logits and isinstance(model.activation, nn.ReLU):
            return "unsafe_relu_logits"
        if isinstance(model.activation, nn.ReLU):
            return "relu_hidden_only"
    return "none"


def build_model(
    *,
    model_type: str,
    in_channels: dict[str, int],
    out_channels: int,
    node_types: Iterable[str],
    relations: Iterable[DirectedRelation],
    target_type: str | None = None,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    relation_gate: bool = False,
    relation_gate_init: float = 1.0,
    block_in_channels: dict[str, int] | None = None,
    block_gate: bool = True,
    block_gate_init: float = 1.0,
    final_activation: str | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    node_type_list = list(node_types)
    relation_list = list(relations)
    if model_type == "relation_linear":
        activation = None
        if final_activation == "relu":
            activation = nn.ReLU()
        elif final_activation not in {None, "none"}:
            raise ValueError("final_activation must be 'none', 'relu', or None")
        model = WeightedRelationLinearConv(
            in_channels=in_channels,
            out_channels=out_channels,
            node_types=node_type_list,
            relations=relation_list,
            activation=activation,
            relation_gate=relation_gate,
            relation_gate_init=relation_gate_init,
        )
    elif model_type == "relation_mlp":
        model = RelationMessageEncoderMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            node_types=node_type_list,
            relations=relation_list,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif model_type == "shadow_fusion":
        if target_type is None:
            raise ValueError("target_type is required for shadow_fusion")
        model = ShadowFusionClassifier(
            in_channels=in_channels,
            out_channels=out_channels,
            node_types=node_type_list,
            relations=relation_list,
            target_type=target_type,
            block_in_channels=block_in_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            relation_gate=relation_gate,
            relation_gate_init=relation_gate_init,
            block_gate=block_gate,
            block_gate_init=block_gate_init,
        )
    else:
        raise ValueError("model_type must be relation_linear, relation_mlp, or shadow_fusion")

    status = final_logits_activation_status(model, used_as_logits=True)
    if status == "unsafe_relu_logits":
        raise ValueError("unsafe final ReLU logits are not allowed")
    diagnostics: dict[str, object] = {"model_type": model_type, "final_logits_activation": status}
    if hasattr(model, "diagnostics"):
        diagnostics.update(model.diagnostics())  # type: ignore[attr-defined]
    return model, diagnostics
