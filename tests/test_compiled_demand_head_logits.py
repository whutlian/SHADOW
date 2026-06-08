from __future__ import annotations

import torch

from shadow_hgc.features.compiled_table import CompiledDemandBlock, compile_demand_table
from shadow_hgc.models.compiled_demand import CompiledDemandMLP


def test_compiled_demand_mlp_returns_raw_logits_without_final_relu():
    table, schema = compile_demand_table(
        [
            (CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.ones(3, 2)),
            (CompiledDemandBlock("degree", "degree", None, 1, "none"), torch.zeros(3, 1)),
        ]
    )
    model = CompiledDemandMLP(
        schema,
        num_classes=2,
        hidden_dim=4,
        dropout=0.0,
        block_norm="none",
        block_gate=False,
        fusion="concat_mlp",
    )

    with torch.no_grad():
        final_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
        final_linear.weight.zero_()
        final_linear.bias.fill_(-0.75)

    logits = model(table)

    assert model.diagnostics()["final_logits_activation"] == "none"
    assert logits.shape == (3, 2)
    assert logits.min().item() < 0.0
    assert torch.allclose(logits, torch.full((3, 2), -0.75))
