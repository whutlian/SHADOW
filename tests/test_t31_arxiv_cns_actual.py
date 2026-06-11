from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from scripts.run_t31_arxiv_actual_cns import build_arxiv_cns_rows
from shadow_hgc.sft.arxiv_cns_actual_v2 import cns_grid_plan, is_historical_lad_predictor


def test_t31_historical_lad_is_diagnostic_not_main() -> None:
    assert is_historical_lad_predictor("ogbn-arxiv_lad_reference_historical_replay_seed42")
    assert not is_historical_lad_predictor("raw_x_mlp")


def test_t31_cns_grid_plan_logs_direction_and_autoscale() -> None:
    plan = cns_grid_plan(correction_alphas=[0.2], smoothing_alphas=[0.4], correction_steps=[10], smoothing_steps=[20], autoscale=["on"], graph_directions=["cite_ref"])
    assert plan[0]["autoscale"] == "on"
    assert plan[0]["graph_direction"] == "cite_ref"


def test_t31_arxiv_cns_missing_base_logits_blocks_without_promotion(tmp_path: Path) -> None:
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
            hidden_dims=[8],
            epochs=1,
            run_long=False,
            device="cpu",
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "missing_base_logits"
    assert rows[0]["promotion_status"] == "not_promoted"
