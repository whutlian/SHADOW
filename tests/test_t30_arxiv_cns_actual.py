from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from scripts.run_t30_arxiv_cns_actual import build_arxiv_cns_rows
from shadow_hgc.sft.arxiv_cns_actual import BaseLogitCache, run_t30_cns_grid


def test_t30_arxiv_cns_missing_logits_blocks_without_smoke_status(tmp_path: Path) -> None:
    rows = build_arxiv_cns_rows(
        Namespace(
            seed=42,
            base_predictors=["raw_x_mlp"],
            base_logits_dir=tmp_path,
            dataset_root=tmp_path / "missing",
            correction_alphas=[0.4],
            smoothing_alphas=[0.4],
            correction_steps=[1],
            smoothing_steps=[1],
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "missing_base_logits"
    assert "smoke" not in rows[0]["status"]


def test_t30_cns_grid_uses_valid_for_selection_not_as_input() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    cache = BaseLogitCache(
        logits=logits,
        metadata={
            "dataset": "ogbn-arxiv",
            "base_predictor": "raw_x_mlp",
            "valid_acc": 1.0,
            "test_acc": 1.0,
            "macro_f1": 1.0,
            "predicted_classes": 2,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
        },
    )
    result = run_t30_cns_grid(
        cache=cache,
        labels=labels,
        train_idx=torch.tensor([0, 1]),
        valid_idx=torch.tensor([2]),
        test_idx=torch.tensor([3]),
        edge_index=edge_index,
        num_classes=2,
        correction_alphas=[0.2],
        smoothing_alphas=[0.2],
        correction_steps=[1],
        smoothing_steps=[1],
    )
    assert result.best_row["status"] == "completed_long"
    assert result.diagnostics["uses_valid_labels_as_input"] is False
    assert result.diagnostics["uses_test_labels_as_input"] is False
    assert result.diagnostics["uses_valid_labels_for_selection"] is True
