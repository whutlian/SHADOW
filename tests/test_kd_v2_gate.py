from __future__ import annotations

from shadow_hgc.teacher.kd_v2 import teacher_quality_gate
from shadow_hgc.models.distill_losses import kd_v2_loss
import torch


def test_kd_v2_gate_rejects_weak_teacher():
    gate = teacher_quality_gate(
        teacher_val_acc=0.44,
        current_best_reference_acc=0.40,
        teacher_predicted_class_count=5,
        num_classes=5,
        teacher_entropy=1.2,
    )

    assert gate.passed is False
    assert gate.reason == "teacher_val_acc_below_reference_margin"


def test_kd_v2_gate_rejects_class_collapse_and_accepts_valid_teacher():
    collapsed = teacher_quality_gate(
        teacher_val_acc=0.80,
        current_best_reference_acc=0.40,
        teacher_predicted_class_count=2,
        num_classes=5,
        teacher_entropy=1.2,
    )
    valid = teacher_quality_gate(
        teacher_val_acc=0.80,
        current_best_reference_acc=0.40,
        teacher_predicted_class_count=5,
        num_classes=5,
        teacher_entropy=1.2,
    )

    assert collapsed.passed is False
    assert collapsed.reason == "teacher_predicted_class_count_too_low"
    assert valid.passed is True


def test_kd_v2_loss_returns_separate_ce_and_kd_terms():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])

    result = kd_v2_loss(student, labels, teacher, temperature=2.0, lambda_kd=0.1)

    assert set(result) == {"loss", "ce_loss", "kd_loss", "kd_to_ce_ratio"}
    assert torch.isfinite(result["loss"])
    assert result["loss"].item() > result["ce_loss"].item()
