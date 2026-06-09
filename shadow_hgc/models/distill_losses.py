from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    weight: float = 1.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shape")
    temperature = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)
    return float(weight) * loss


def ce_kd_loss(
    student_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    kd_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, hard_labels.to(torch.long))
    kd = kd_kl_loss(student_logits, teacher_logits, temperature=temperature, weight=kd_weight)
    return float(ce_weight) * ce + kd
