from __future__ import annotations

import torch

from shadow_hgc.sft.stt_training import stt_soft_target_loss


def test_t34_stt_loss_uses_soft_hard_prior_and_calibration_terms() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    y_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    y_hard = torch.tensor([0, 1])
    hard_mask = torch.tensor([True, False])
    prior = torch.tensor([0.5, 0.5])
    loss, parts = stt_soft_target_loss(
        logits,
        y_soft,
        y_hard=y_hard,
        hard_anchor_mask=hard_mask,
        teacher_prior=prior,
        temperature=2.0,
        lambda_soft=1.0,
        lambda_hard=0.25,
        lambda_prior=0.02,
        lambda_calib=0.02,
    )
    assert loss.item() > 0.0
    assert parts["soft_loss"] > 0.0
    assert parts["hard_loss"] >= 0.0
    assert parts["calib_loss"] >= 0.0


def test_t34_virtual_mixup_loss_does_not_change_row_count() -> None:
    logits = torch.randn(4, 3)
    y_soft = torch.softmax(torch.randn(4, 3), dim=1)
    loss, parts = stt_soft_target_loss(logits, y_soft, lambda_mix=0.05, mixup_alpha=0.4, seed=3)
    assert loss.item() > 0.0
    assert parts["virtual_mixup_count"] == 4
