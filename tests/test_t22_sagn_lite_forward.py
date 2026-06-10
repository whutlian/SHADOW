from __future__ import annotations

import torch

from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3


def test_sagn_lite_v2_forward_logs_attention_and_label_branch():
    blocks = {
        "self": torch.randn(5, 4),
        "X1": torch.randn(5, 4),
        "Y1": torch.randn(5, 3),
        "structure": torch.randn(5, 2),
    }
    model = SFTTeacherV3(
        {name: value.shape[1] for name, value in blocks.items()},
        num_classes=3,
        model_type="sagn_lite_v2",
        hidden_dim=16,
        dropout=0.0,
        block_dropout=0.1,
        hop_dropout=0.05,
        norm="layernorm",
    )
    model.fit_block_stats(blocks, train_rows=torch.tensor([0, 1, 2]))
    logits = model(blocks)
    diag = model.diagnostics()

    assert logits.shape == (5, 3)
    assert diag["model_type"] == "sagn_lite_v2"
    assert diag["has_label_branch"] is True
    assert diag["final_logits_activation"] == "none"
    assert set(diag["block_gates"]) == set(blocks)
