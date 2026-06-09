from __future__ import annotations

import torch
import pytest

from shadow_hgc.fullgraph.robust_block_selection import (
    build_stratified_folds,
    robust_keep_decision,
    selection_score,
)


def test_stratified_3fold_keeps_candidate_with_better_median_and_no_bad_regression():
    base = [
        {"valid_acc": 0.50, "valid_macro_f1": 0.40, "class_coverage": 0.70},
        {"valid_acc": 0.51, "valid_macro_f1": 0.42, "class_coverage": 0.70},
        {"valid_acc": 0.49, "valid_macro_f1": 0.39, "class_coverage": 0.70},
    ]
    candidate = [
        {"valid_acc": 0.53, "valid_macro_f1": 0.44, "class_coverage": 0.90},
        {"valid_acc": 0.52, "valid_macro_f1": 0.43, "class_coverage": 0.90},
        {"valid_acc": 0.50, "valid_macro_f1": 0.40, "class_coverage": 0.90},
    ]
    decision = robust_keep_decision(base, candidate, tolerance=0.02)

    assert selection_score(valid_acc=0.5, valid_macro_f1=0.4, class_coverage=0.7) == pytest.approx(0.615)
    assert decision.keep is True
    assert decision.protocol == "stratified_3fold"
    assert decision.median_candidate_score > decision.median_base_score


def test_stratified_3fold_uses_only_train_and_valid_rows():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    train_rows = torch.tensor([0, 1, 3, 4, 6, 7], dtype=torch.long)
    valid_rows = torch.tensor([2, 5, 8], dtype=torch.long)
    test_rows = torch.tensor([9], dtype=torch.long)
    folds = build_stratified_folds(labels, train_rows=train_rows, valid_rows=valid_rows, test_rows=test_rows, k=3, seed=42)

    used = set()
    for fold in folds:
        used.update(fold.valid_rows.tolist())
        assert not set(fold.valid_rows.tolist()).intersection(set(test_rows.tolist()))
    assert used.issubset(set(train_rows.tolist()) | set(valid_rows.tolist()))
