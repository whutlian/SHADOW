from __future__ import annotations

import torch

from shadow_hgc.training.two_stage import TwoStageConfig, train_sft_two_stage


def test_two_stage_config_expands_loss_schedule():
    config = TwoStageConfig(
        enabled=True,
        stage1_loss="sqrt_weighted_ce",
        stage2_loss="cross_entropy",
        stage1_epochs=3,
        stage2_epochs=2,
        stage2_lr_mult=0.2,
    )

    assert [stage.loss_type for stage in config.stages()] == ["sqrt_weighted_ce", "cross_entropy"]
    assert [stage.epochs for stage in config.stages()] == [3, 2]
    assert [stage.lr_mult for stage in config.stages()] == [1.0, 0.2]


def test_two_stage_training_runs_without_logits_or_kd():
    blocks = {"self": torch.randn(12, 4), "X1": torch.randn(12, 4), "Y1": torch.randn(12, 3)}
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
    result = train_sft_two_stage(
        blocks=blocks,
        labels=labels,
        train_rows=torch.arange(0, 8),
        valid_rows=torch.arange(8, 10),
        test_rows=torch.arange(10, 12),
        num_classes=3,
        model_type="sagn_lite_v2",
        hidden_dim=8,
        config=TwoStageConfig(enabled=True, stage1_epochs=1, stage2_epochs=1),
        batch_size=None,
        seed=42,
    )

    assert result.summary["stages"][0]["loss_type"] == "sqrt_weighted_ce"
    assert result.summary["stages"][1]["loss_type"] == "cross_entropy"
    assert result.summary["uses_logits_as_input"] is False
    assert result.summary["uses_kd"] is False
