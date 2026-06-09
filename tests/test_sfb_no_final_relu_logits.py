from __future__ import annotations

import torch

from shadow_hgc.fullgraph.sfb_model import BlockGatedResidualTableModel


def test_sfb_outputs_raw_logits_without_final_relu():
    blocks = {"self": torch.randn(5, 3)}
    model = BlockGatedResidualTableModel({"self": 3}, num_classes=2, hidden_dim=4)
    model.fit_block_stats(blocks, source="train_target_rows")
    model.freeze_block_stats()
    with torch.no_grad():
        model.logit_heads["self"].bias.fill_(-5.0)

    logits = model(blocks)

    assert (logits < 0).any()
    assert model.diagnostics()["final_logits_activation"] == "none"
