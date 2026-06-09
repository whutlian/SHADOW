from __future__ import annotations

import torch

from shadow_hgc.features.compiled_table import CompiledDemandBlock, compile_demand_table
from shadow_hgc.models.compiled_demand import CompiledDemandMLP, fit_compiled_block_stats


def test_compiled_block_stats_do_not_change_after_condensed_training_forward():
    train_full_table, schema = compile_demand_table(
        [(CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[0.0, 2.0], [4.0, 10.0]]))]
    )
    condensed_table, condensed_schema = compile_demand_table(
        [(CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[100.0, 200.0], [300.0, 400.0]]))]
    )
    assert schema == condensed_schema
    model = CompiledDemandMLP(
        schema,
        num_classes=2,
        hidden_dim=4,
        dropout=0.0,
        block_norm="standardize",
        block_gate=False,
        lazy_block_stats=False,
    )
    fit_compiled_block_stats(model, train_full_table, schema)
    before = model.block_norm_stats()["self"]

    model.train()
    _ = model(condensed_table)
    after_train = model.block_norm_stats()["self"]
    model.eval()
    _ = model(train_full_table)
    after_infer = model.block_norm_stats()["self"]

    assert after_train["mean"] == before["mean"]
    assert after_train["std"] == before["std"]
    assert after_infer["mean"] == before["mean"]
    assert after_infer["std"] == before["std"]
    assert after_train["frozen"] is True
