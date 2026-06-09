from __future__ import annotations

import torch

from shadow_hgc.models.sehgnn_lite import SeHGNNLite


def test_sehgnn_lite_forward_uses_named_blocks_and_returns_logits():
    blocks = {
        "self": torch.randn(5, 3),
        "MAM": torch.randn(5, 4),
        "PathLAD-MAM": torch.randn(5, 2),
    }
    model = SeHGNNLite(
        block_dims={"self": 3, "MAM": 4, "PathLAD-MAM": 2},
        num_classes=3,
        hidden_dim=8,
        dropout=0.0,
        block_norm="standardize",
        block_gate=True,
        fusion="concat_mlp",
        lazy_block_stats=False,
    )
    model.fit_block_stats(blocks, source="train_full_target_rows")
    model.freeze_block_stats()

    logits = model(blocks)

    assert logits.shape == (5, 3)
    assert set(model.block_gate_values()) == {"self", "MAM", "PathLAD-MAM"}
    assert model.diagnostics()["block_norm_stats_source"] == "train_full_target_rows"
