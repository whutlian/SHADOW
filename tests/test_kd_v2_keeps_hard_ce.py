from __future__ import annotations

import torch
import torch.nn.functional as F

from shadow_hgc.models.distill_losses import kd_v2_loss


def test_kd_v2_keeps_hard_ce_when_lambda_is_zero():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1])

    result = kd_v2_loss(student, labels, teacher, temperature=2.0, lambda_kd=0.0)

    assert torch.allclose(result["loss"], F.cross_entropy(student, labels))
    assert torch.allclose(result["ce_loss"], F.cross_entropy(student, labels))
