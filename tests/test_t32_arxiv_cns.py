from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_t32_arxiv_actual_cns import build_arxiv_cns_rows
from shadow_hgc.sft.arxiv_base_predictors_v3 import validate_arxiv_split_and_feature_alignment
from shadow_hgc.sft.t32_arxiv_cns import cns_failure_reason, cns_grid_plan_v2


def test_t32_cns_grid_logs_direction_normalization_and_self_loop() -> None:
    plan = cns_grid_plan_v2(
        graph_directions=["cite_ref"],
        correction_alphas=[0.2],
        smoothing_alphas=[0.4],
        correction_steps=[10],
        smoothing_steps=[20],
        autoscale=["on"],
        normalization_modes=["dst_row"],
        self_loop_modes=["none"],
    )
    assert plan[0]["graph_direction"] == "cite_ref"
    assert plan[0]["normalization_mode"] == "dst_row"
    assert plan[0]["self_loop_mode"] == "none"


def test_t32_weak_cns_row_is_marked_pipeline_mismatch() -> None:
    assert cns_failure_reason(cns_accuracy=0.64, base_predictor="raw_x_mlp") == "cns_pipeline_mismatch_or_weak_base"
    assert cns_failure_reason(cns_accuracy=0.72, base_predictor="raw_x_mlp") == ""


def test_t32_arxiv_missing_base_logits_blocks_without_historical_fallback(tmp_path: Path) -> None:
    rows = build_arxiv_cns_rows(
        Namespace(
            seed=42,
            base_predictors=["raw_x_mlp"],
            base_logits_dir=str(tmp_path),
            dataset_root=str(tmp_path / "missing"),
            train_base_logits_if_missing=False,
            correction_alphas=[0.2],
            smoothing_alphas=[0.4],
            correction_steps=[1],
            smoothing_steps=[1],
            autoscale=["off"],
            graph_directions=["cite_ref"],
            normalization_modes=["dst_row"],
            self_loop_modes=["none"],
            hidden_dims=[8],
            epochs=1,
            run_long=False,
            device="cpu",
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "missing_base_logits"
    assert rows[0]["base_logit_cache_path"] == ""


def test_t32_arxiv_alignment_missing_dataset_blocks(tmp_path: Path) -> None:
    diag = validate_arxiv_split_and_feature_alignment(tmp_path / "missing")
    assert diag["blocked"] is True
    assert diag["failure_reason"] == "missing_arxiv_dataset"
