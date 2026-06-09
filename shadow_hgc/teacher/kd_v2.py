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

