from __future__ import annotations

from shadow_hgc.audit.parity import validate_promoted_row


def test_promoted_full_schema_row_requires_schema_edges():
    row = {
        "dataset": "dblp",
        "variant": "S1_clean_full_schema_sehgnn",
        "status": "completed",
        "loader_mode": "full_schema",
        "schema_required_edges_present": False,
        "model_type": "sehgnn_lite",
        "metapath_blocks": ["APA"],
    }

    checks = validate_promoted_row(row)

    assert checks["valid"] is False
    assert "full_schema_required_edges_absent" in checks["reasons"]


def test_promoted_sehgnn_metapath_row_requires_real_blocks():
    checks = validate_promoted_row(
        {
            "dataset": "acm",
            "variant": "S1_clean_metapath_sehgnn",
            "status": "completed",
            "model_type": "compiled_demand_mlp",
            "metapath_blocks": [],
        }
    )

    assert checks["valid"] is False
    assert "sehgnn_row_requires_model_type_sehgnn_lite" in checks["reasons"]
    assert "metapath_blocks_missing" in checks["reasons"]
