from __future__ import annotations

import torch

from shadow_hgc.features.compiled_table import CompiledDemandBlock, compile_demand_table
from shadow_hgc.models.compiled_demand import apply_block_stats, fit_block_stats, freeze_block_stats


def test_compiled_block_stats_fit_freeze_metadata_has_required_fields():
    table, schema = compile_demand_table(
        [
            (CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.tensor([[1.0, 3.0], [3.0, 7.0]])),
            (CompiledDemandBlock("degree", "degree", None, 1, "standardize"), torch.tensor([[2.0], [6.0]])),
        ]
    )

    stats = freeze_block_stats(fit_block_stats(table, schema, source="train_full_target_demand_table"))
    normalized, metadata = apply_block_stats(table, schema, stats)

    assert normalized.shape == table.shape
    assert metadata["block_norm_stats_source"] == "train_full_target_demand_table"
    assert metadata["block_names"] == ["self", "degree"]
    assert metadata["block_dims"] == {"self": 2, "degree": 1}
    assert metadata["stats_fit_num_rows"] == 2
    assert metadata["stats_frozen"] is True
    assert "self" in metadata["block_mean_norms"]
    assert metadata["block_std_min"] > 0.0
    assert metadata["block_std_max"] >= metadata["block_std_min"]
