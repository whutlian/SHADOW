from __future__ import annotations

import torch

from shadow_hgc.features.compiled_table import CompiledDemandBlock, compile_demand_table
from shadow_hgc.models.compiled_demand import CompiledDemandMLP, fit_compiled_block_stats


def test_fit_compiled_block_stats_uses_train_full_table_source():
    train_full_table, schema = compile_demand_table(
        [
            (CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[1.0, 3.0], [3.0, 7.0]])),
            (CompiledDemandBlock("degree", "degree", None, 1, "standardize"), torch.tensor([[2.0], [6.0]])),
        ]
    )
    model = CompiledDemandMLP(
        schema,
        num_classes=2,
        hidden_dim=4,
        dropout=0.0,
        block_norm="standardize",
        block_gate=False,
        lazy_block_stats=False,
    )

    metadata = fit_compiled_block_stats(model, train_full_table, schema)

    stats = model.block_norm_stats()
    assert metadata["compiled_block_stats_source"] == "train_full_demand_table"
    assert stats["self"]["source"] == "train_full_demand_table"
    assert stats["degree"]["source"] == "train_full_demand_table"
    assert stats["self"]["frozen"] is True
    assert stats["degree"]["frozen"] is True
    assert stats["self"]["mean"] == [2.0, 5.0]
    assert stats["degree"]["mean"] == [4.0]


def test_lazy_block_stats_disabled_requires_explicit_fit():
    table, schema = compile_demand_table(
        [(CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.ones(3, 2))]
    )
    model = CompiledDemandMLP(
        schema,
        num_classes=2,
        hidden_dim=4,
        dropout=0.0,
        block_norm="standardize",
        block_gate=False,
        lazy_block_stats=False,
    )

    try:
        model(table)
    except RuntimeError as exc:
        assert "block stats must be fitted" in str(exc)
    else:
        raise AssertionError("forward should reject unfitted frozen block stats")
