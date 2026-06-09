from __future__ import annotations

import torch

from shadow_hgc.fullgraph.sfb_model import BlockGatedResidualTableModel


def test_sfb_block_stats_fit_on_train_rows_and_freeze():
    blocks = {"self": torch.randn(6, 3), "scap": torch.randn(6, 2)}
    model = BlockGatedResidualTableModel({"self": 3, "scap": 2}, num_classes=4, hidden_dim=8)

    stats = model.fit_block_stats({name: value[:3] for name, value in blocks.items()}, source="train_target_rows")
    model.freeze_block_stats()
    logits = model(blocks)

    assert logits.shape == (6, 4)
    assert stats["block_norm_stats_source"] == "train_target_rows"
    assert model.diagnostics()["stats_frozen"] is True
    assert model.diagnostics()["stats_fit_num_rows"] == 3
