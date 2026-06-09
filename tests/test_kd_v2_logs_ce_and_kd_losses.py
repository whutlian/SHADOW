from __future__ import annotations

from shadow_hgc.teacher.kd_v2 import kd_v2_log_summary


def test_kd_v2_log_summary_contains_required_loss_fields():
    row = kd_v2_log_summary(
        teacher_type="sehgnn_lite",
        teacher_train_acc=0.95,
        teacher_val_acc=0.90,
        teacher_predicted_class_count=3,
        teacher_entropy=1.0,
        student_predicted_class_count=3,
        temperature=2.0,
        lambda_kd=0.1,
        ce_losses=[2.0, 1.0],
        kd_losses=[0.5, 0.2],
        warmup_epochs=100,
        kd_gate_passed=True,
        kd_skip_reason="",
    )

    assert row["ce_loss_start"] == 2.0
    assert row["ce_loss_end"] == 1.0
    assert row["kd_loss_start"] == 0.5
    assert row["kd_loss_end"] == 0.2
    assert row["kd_to_ce_ratio"] == 0.2
