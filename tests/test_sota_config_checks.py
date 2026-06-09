from __future__ import annotations

from shadow_hgc.audit.config_checks import assert_or_mark_invalid, validate_variant_config


def test_sehgnn_metapath_variant_requires_real_sehgnn_blocks_and_stats():
    checks = validate_variant_config(
        {
            "variant": "S1_sehgnn_lite_metapath",
            "model_type": "compiled_demand_mlp",
            "metapath_blocks": ["PAP"],
            "block_dims": {"PAP": 16},
            "block_norm_stats_source": "train_full_target_rows",
            "feature_blocks": ["self", "PAP"],
        },
        dataset_meta={},
        runtime_diagnostics={},
    )

    assert checks["valid"] is False
    assert "sehgnn_or_metapath_requires_model_type_sehgnn_lite" in checks["reasons"]


def test_kd_variant_requires_teacher_quality_and_separate_losses():
    checks = validate_variant_config(
        {
            "variant": "S4_teacher_kd",
            "model_type": "sehgnn_lite",
            "teacher_type": "sehgnn_lite",
            "teacher_train_acc": 0.91,
            "teacher_val_acc": 0.88,
            "teacher_predicted_class_count": 3,
            "num_classes": 3,
            "kd_lambda": 0.1,
            "temperature": 2.0,
            "ce_loss": 0.8,
        },
        dataset_meta={},
        runtime_diagnostics={},
    )

    assert checks["valid"] is False
    assert "kd_loss_missing" in checks["reasons"]


def test_invalid_row_is_marked_and_metrics_are_cleared():
    checks = {"valid": False, "reasons": ["missing_blocks"], "warnings": []}
    row = assert_or_mark_invalid({"status": "completed", "accuracy": 0.9, "macro_f1": 0.8}, checks)

    assert row["status"] == "invalid_config"
    assert row["accuracy"] is None
    assert row["macro_f1"] is None
    assert row["invalid_reasons"] == ["missing_blocks"]
