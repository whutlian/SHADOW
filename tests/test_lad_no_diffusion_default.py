from __future__ import annotations

from scripts.run_lad_common import LAD_STAGE_DEFAULTS, lad_feature_mode


def test_lad_defaults_disable_diffusion() -> None:
    assert LAD_STAGE_DEFAULTS["diffusion_enabled"] is False
    assert LAD_STAGE_DEFAULTS["diffusion_status"] == "diagnostic_only"
    assert LAD_STAGE_DEFAULTS["feature_mode"] == "label_affinity"


def test_lad_feature_modes_are_not_diffusion() -> None:
    for use_lad in (False, True):
        for use_metapath in (False, True):
            mode = lad_feature_mode(label_affinity=use_lad, metapath=use_metapath)
            assert "diffusion" not in mode
