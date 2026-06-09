from __future__ import annotations

from shadow_hgc.features.scap_v2 import validate_scap_v2_config


def test_scap_v2_rejects_dense_p2_promoted_config():
    result = validate_scap_v2_config({"uses_dense_p2": True, "uses_high_dim_diffusion": False})

    assert result["valid"] is False
    assert "uses_dense_p2" in result["invalid_reasons"]
