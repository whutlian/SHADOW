from __future__ import annotations

from shadow_hgc.teacher.kd_v2 import teacher_quality_gate


def test_kd_v2_gate_requires_teacher_margin_over_current_best():
    gate = teacher_quality_gate(
        teacher_val_acc=0.69,
        current_best_reference_acc=0.66,
        teacher_predicted_class_count=40,
        num_classes=40,
        teacher_entropy=2.0,
    )

    assert gate.passed is False
    assert gate.reason == "teacher_val_acc_below_reference_margin"
