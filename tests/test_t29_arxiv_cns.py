from __future__ import annotations

import argparse

import torch

from scripts.run_t29_arxiv_cns_actual import build_arxiv_cns_rows
from shadow_hgc.sft.arxiv_actual_cns import MissingBaseLogitsError, require_base_logits, run_actual_cns_grid


def _toy_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [3.0, 0.2, 0.1],
            [0.1, 2.5, 0.0],
            [0.1, 0.2, 2.8],
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.1],
        ],
        dtype=torch.float32,
    )


def test_t29_cns_requires_base_logits():
    try:
        require_base_logits(None)
    except MissingBaseLogitsError as exc:
        assert "missing_base_logits" in str(exc)
    else:
        raise AssertionError("require_base_logits must reject missing logits")

    args = argparse.Namespace(seed=42, base_predictors=["raw_x_mlp"], base_logits_dir="missing", smoke=False)
    rows = build_arxiv_cns_rows(args)
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "missing_base_logits"


def test_t29_cns_no_test_label_leakage():
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    labels_a = torch.tensor([0, 1, 2, 0, 1])
    labels_b = torch.tensor([0, 1, 2, 2, 0])
    common = dict(
        logits=_toy_logits(),
        train_idx=torch.tensor([0, 1, 2]),
        valid_idx=torch.tensor([3]),
        test_idx=torch.tensor([4]),
        edge_index=edge_index,
        num_classes=3,
        correction_alphas=[0.4],
        smoothing_alphas=[0.4],
        correction_steps=[2],
        smoothing_steps=[2],
    )
    out_a = run_actual_cns_grid(labels=labels_a, **common)
    out_b = run_actual_cns_grid(labels=labels_b, **common)
    assert torch.allclose(out_a.best_probs, out_b.best_probs)
    assert out_a.diagnostics["uses_test_labels_as_input"] is False


def test_t29_cns_valid_labels_not_features():
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 0, 1])
    result = run_actual_cns_grid(
        logits=_toy_logits(),
        labels=labels,
        train_idx=torch.tensor([0, 1, 2]),
        valid_idx=torch.tensor([3]),
        test_idx=torch.tensor([4]),
        edge_index=edge_index,
        num_classes=3,
        correction_alphas=[0.2, 0.4],
        smoothing_alphas=[0.2],
        correction_steps=[1],
        smoothing_steps=[1],
    )
    assert result.best_row["status"] == "completed_real"
    assert result.diagnostics["uses_valid_labels_as_input"] is False
    assert result.diagnostics["uses_valid_labels_for_selection"] is True


def test_t29_arxiv_teacher_gate_blocks_condensation_below_a1():
    args = argparse.Namespace(seed=42, base_predictors=["raw_x_mlp"], base_logits_dir="missing", smoke=False)
    row = build_arxiv_cns_rows(args)[0]
    assert row["promotion_status"] == "not_promoted"
    assert row["teacher_accuracy"] == ""
