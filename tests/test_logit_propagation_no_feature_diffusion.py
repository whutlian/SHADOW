from __future__ import annotations

from shadow_hgc.features.logit_propagation import validate_logit_propagation_config


def test_logit_propagation_rejects_feature_diffusion():
    result = validate_logit_propagation_config(
        num_classes=3,
        input_dim=128,
        propagates_features=True,
    )

    assert result["valid"] is False
    assert "feature_diffusion_forbidden" in result["invalid_reasons"]
