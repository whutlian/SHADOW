from __future__ import annotations

import pytest
import torch

from shadow_hgc.models.sft_table_teacher import SFTTableTeacherV2


def test_sft_table_teacher_v2_returns_raw_logits_for_residual_block_gated():
    blocks = {
        "self": torch.randn(6, 3),
        "typed:cite": torch.randn(6, 2),
    }
    model = SFTTableTeacherV2(
        {"self": 3, "typed:cite": 2},
        num_classes=4,
        model_type="residual_block_gated",
        hidden_dim=8,
        dropout=0.0,
    )
    model.fit_block_stats(blocks, train_rows=torch.tensor([0, 1, 2], dtype=torch.long))
    logits = model(blocks)

    assert logits.shape == (6, 4)
    assert logits.requires_grad
    assert model.diagnostics()["final_logits_activation"] == "none"
    assert torch.any(logits < 0) or torch.any(logits > 1)


def test_sft_table_teacher_v2_rejects_logit_or_kd_block_names():
    with pytest.raises(ValueError, match="logits are forbidden"):
        SFTTableTeacherV2({"teacher_logits": 3}, num_classes=2)
    with pytest.raises(ValueError, match="logits are forbidden"):
        SFTTableTeacherV2({"kd_soft": 3}, num_classes=2)
