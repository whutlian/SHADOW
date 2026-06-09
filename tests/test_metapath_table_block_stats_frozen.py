from __future__ import annotations

import torch

from shadow_hgc.models.block_gated_table import BlockGatedTableModel


def test_metapath_table_block_stats_are_fit_on_train_rows_and_frozen():
    blocks = {"self": torch.randn(5, 4), "metapath:PAP": torch.randn(5, 3)}
    model = BlockGatedTableModel({"self": 4, "metapath:PAP": 3}, num_classes=2, hidden_dim=8)

    model.fit_block_stats({name: value[:3] for name, value in blocks.items()}, source="train_target_rows")
    model.freeze_block_stats()

    diagnostics = model.diagnostics()
    assert diagnostics["block_norm_stats_source"] == "train_target_rows"
    assert diagnostics["stats_frozen"] is True
    assert diagnostics["stats_fit_num_rows"] == 3
