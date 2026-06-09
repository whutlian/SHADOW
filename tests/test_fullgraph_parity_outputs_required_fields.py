from __future__ import annotations

from shadow_hgc.audit.parity import FULLGRAPH_PARITY_REQUIRED_FIELDS, validate_fullgraph_parity_row


def _valid_row() -> dict:
    return {
        field: "x"
        for field in FULLGRAPH_PARITY_REQUIRED_FIELDS
    } | {
        "dataset": "acm",
        "variant": "fullgraph_sehgnn_lite_current",
        "seed": 42,
        "status": "completed",
        "accuracy": 0.91,
        "macro_f1": 0.90,
        "weighted_f1": 0.90,
        "predicted_class_count": 3,
        "target_gate": 0.90,
        "gate_passed": True,
        "blocked_by_fullgraph_backbone": False,
        "train_nodes": 10,
        "valid_nodes": 0,
        "test_nodes": 20,
        "num_classes": 3,
        "train_class_count": 3,
        "valid_class_count": 0,
        "test_class_count": 3,
        "training_time_s": 1.0,
        "inference_time_s": 0.1,
        "peak_cpu_ram_mb": 100.0,
        "peak_gpu_ram_mb": 0.0,
    }


def test_fullgraph_parity_row_has_required_fields_and_hashes():
    checks = validate_fullgraph_parity_row(_valid_row())

    assert checks["valid"] is True
    assert checks["reasons"] == []


def test_fullgraph_parity_row_missing_hashes_is_invalid():
    row = _valid_row()
    row["split_hash"] = ""
    row.pop("feature_hash")

    checks = validate_fullgraph_parity_row(row)

    assert checks["valid"] is False
    assert "split_hash_missing" in checks["reasons"]
    assert "feature_hash_missing" in checks["reasons"]
