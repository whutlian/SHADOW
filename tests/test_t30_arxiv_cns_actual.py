from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts.run_t30_arxiv_cns_actual import build_arxiv_cns_rows
from shadow_hgc.sft.arxiv_cns_actual import BaseLogitCache, run_t30_cns_grid
from shadow_hgc.sft.arxiv_logits import load_base_logit_cache


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


def test_t30_arxiv_loader_accepts_all_target_memmap_cache(tmp_path: Path) -> None:
    logits_path = tmp_path / "all_target_logits.memmap"
    mm = np.memmap(logits_path, mode="w+", dtype=np.float32, shape=(3, 2))
    mm[:] = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    mm.flush()
    (tmp_path / "meta.json").write_text(
        __import__("json").dumps(
            {
                "meta": {"accuracy": 0.75, "macro_f1": 0.5, "predicted_class_count": 2},
                "storage": {
                    "all_target_logits": {
                        "file": logits_path.name,
                        "dtype": "float32",
                        "shape": [3, 2],
                        "storage": "memmap",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cache = load_base_logit_cache(tmp_path)
    assert tuple(cache.logits.shape) == (3, 2)
    assert cache.metadata["test_acc"] == 0.75
    assert cache.metadata["macro_f1"] == 0.5


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
