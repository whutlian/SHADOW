from __future__ import annotations

import torch

from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3


def test_gamlp_recursive_v2_forward_uses_self_residual_and_gates():
    blocks = {
        "self": torch.randn(6, 4),
        "X1": torch.randn(6, 4),
        "X2": torch.randn(6, 4),
        "Y1": torch.randn(6, 3),
    }
    model = SFTTeacherV3(
        {name: value.shape[1] for name, value in blocks.items()},
        num_classes=3,
        model_type="gamlp_recursive_v2",
        hidden_dim=12,
        dropout=0.0,
        activation="gelu",
    )
    model.fit_block_stats(blocks, train_rows=torch.tensor([0, 2, 4]))
    logits = model(blocks)
    diag = model.diagnostics()

    assert logits.shape == (6, 3)
    assert diag["model_type"] == "gamlp_recursive_v2"
    assert set(diag["block_gates"]) == {"X1", "X2", "Y1"}
    assert diag["uses_logits_as_input"] is False
