from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KDGateResult:
    passed: bool
    reason: str
    diagnostics: dict


def teacher_quality_gate(
    *,
    teacher_val_acc: float,
    current_best_reference_acc: float,
    teacher_predicted_class_count: int,
    num_classes: int,
    teacher_entropy: float,
    required_margin: float = 0.05,
    min_class_fraction: float = 0.8,
    min_entropy: float = 1e-4,
) -> KDGateResult:
    diagnostics = {
        "teacher_val_acc": float(teacher_val_acc),
        "current_best_reference_acc": float(current_best_reference_acc),
        "teacher_predicted_class_count": int(teacher_predicted_class_count),
        "num_classes": int(num_classes),
        "teacher_entropy": float(teacher_entropy),
        "required_margin": float(required_margin),
        "min_class_fraction": float(min_class_fraction),
    }
    if float(teacher_val_acc) < float(current_best_reference_acc) + float(required_margin):
        return KDGateResult(False, "teacher_val_acc_below_reference_margin", diagnostics)
    if int(teacher_predicted_class_count) < int(float(min_class_fraction) * int(num_classes)):
        return KDGateResult(False, "teacher_predicted_class_count_too_low", diagnostics)
    if float(teacher_entropy) <= float(min_entropy) and int(teacher_predicted_class_count) < int(num_classes):
        return KDGateResult(False, "teacher_entropy_collapse", diagnostics)
    return KDGateResult(True, "passed", diagnostics)


def kd_v2_log_summary(
    *,
    teacher_type: str,
    teacher_train_acc: float,
    teacher_val_acc: float,
    teacher_predicted_class_count: int,
    teacher_entropy: float,
    student_predicted_class_count: int,
    temperature: float,
    lambda_kd: float,
    ce_losses: list[float],
    kd_losses: list[float],
    warmup_epochs: int,
    kd_gate_passed: bool,
    kd_skip_reason: str,
) -> dict:
    ce_start = float(ce_losses[0]) if ce_losses else 0.0
    ce_end = float(ce_losses[-1]) if ce_losses else 0.0
    kd_start = float(kd_losses[0]) if kd_losses else 0.0
    kd_end = float(kd_losses[-1]) if kd_losses else 0.0
    return {
        "teacher_type": teacher_type,
        "teacher_train_acc": float(teacher_train_acc),
        "teacher_val_acc": float(teacher_val_acc),
        "teacher_predicted_class_count": int(teacher_predicted_class_count),
        "teacher_entropy": float(teacher_entropy),
        "student_predicted_class_count": int(student_predicted_class_count),
        "temperature": float(temperature),
        "lambda_kd": float(lambda_kd),
        "ce_loss_start": ce_start,
        "ce_loss_end": ce_end,
        "kd_loss_start": kd_start,
        "kd_loss_end": kd_end,
        "kd_to_ce_ratio": kd_end / max(ce_end, 1e-12),
        "warmup_epochs": int(warmup_epochs),
        "kd_gate_passed": bool(kd_gate_passed),
        "kd_skip_reason": kd_skip_reason,
    }
