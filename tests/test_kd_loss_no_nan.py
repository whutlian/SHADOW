from __future__ import annotations

import torch

from shadow_hgc.models.distill_losses import kd_kl_loss


def test_kd_loss_is_finite_for_extreme_logits():
    student = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    teacher = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]])

    loss = kd_kl_loss(student, teacher, temperature=2.0, weight=0.5)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
