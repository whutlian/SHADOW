from __future__ import annotations

import torch

from scripts.run_t38_arxiv_unified_stage import _classwise_sqrt_budget, select_arxiv_unified_rows


def test_t38_arxiv_classwise_budget_keeps_active_classes() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    train_rows = torch.arange(labels.numel())

    budget = _classwise_sqrt_budget(labels, train_rows, total_budget=6, num_classes=3)

    assert set(budget) == {0, 1, 2}
    assert sum(budget.values()) == 6
    assert min(budget.values()) >= 1


def test_t38_arxiv_selector_signature_has_no_valid_or_test_inputs() -> None:
    args = select_arxiv_unified_rows.__annotations__

    assert "valid_rows" not in args
    assert "test_rows" not in args
    assert "valid_labels" not in args
    assert "test_labels" not in args
