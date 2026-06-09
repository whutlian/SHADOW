from __future__ import annotations

import torch

from shadow_hgc.features.compiled_table import CompiledDemandBlock, compile_demand_table
from shadow_hgc.models.compiled_demand import apply_block_stats, fit_block_stats, freeze_block_stats


def test_same_compiled_block_stats_apply_to_condensed_and_full_tables():
    full, schema = compile_demand_table(
        [(CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[0.0, 2.0], [4.0, 10.0]]))]
    )
    condensed, condensed_schema = compile_demand_table(
        [(CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[100.0, 200.0]]))]
    )

    stats = freeze_block_stats(fit_block_stats(full, schema))
    full_norm, full_meta = apply_block_stats(full, schema, stats)
    condensed_norm, condensed_meta = apply_block_stats(condensed, condensed_schema, stats)

    assert full_meta["stats_frozen"] is True
    assert condensed_meta["stats_frozen"] is True
    assert full_meta["block_mean_norms"] == condensed_meta["block_mean_norms"]
    assert full_norm.shape == full.shape
    assert condensed_norm.shape == condensed.shape
