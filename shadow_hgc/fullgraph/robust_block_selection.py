from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import torch


@dataclass(frozen=True)
class StratifiedFold:
    train_rows: torch.Tensor
    valid_rows: torch.Tensor


@dataclass(frozen=True)
class RobustBlockDecision:
    keep: bool
    protocol: str
    median_base_score: float
    median_candidate_score: float
    worst_regression: float
    reason: str


def selection_score(*, valid_acc: float, valid_macro_f1: float, class_coverage: float) -> float:
    return float(valid_acc) + 0.2 * float(valid_macro_f1) + 0.05 * float(class_coverage)


def _row_score(row: dict[str, Any]) -> float:
    return selection_score(
        valid_acc=float(row.get("valid_acc", row.get("accuracy", 0.0))),
        valid_macro_f1=float(row.get("valid_macro_f1", row.get("macro_f1", 0.0))),
        class_coverage=float(row.get("class_coverage", 0.0)),
    )


def robust_keep_decision(
    base_folds: list[dict[str, Any]],
    candidate_folds: list[dict[str, Any]],
    *,
    tolerance: float = 0.01,
    protocol: str = "stratified_3fold",
) -> RobustBlockDecision:
    if len(base_folds) != len(candidate_folds):
        raise ValueError("base_folds and candidate_folds must have the same length")
    if not base_folds:
        raise ValueError("at least one fold is required")
    base_scores = [_row_score(row) for row in base_folds]
    cand_scores = [_row_score(row) for row in candidate_folds]
    regressions = [base - cand for base, cand in zip(base_scores, cand_scores)]
    med_base = float(median(base_scores))
    med_cand = float(median(cand_scores))
    worst_regression = float(max(regressions))
    keep = med_cand > med_base and worst_regression <= float(tolerance)
    return RobustBlockDecision(
        keep=bool(keep),
        protocol=str(protocol),
        median_base_score=med_base,
        median_candidate_score=med_cand,
        worst_regression=worst_regression,
        reason="median_improved_no_bad_regression" if keep else "median_not_improved_or_regressed",
    )


def build_stratified_folds(
    labels: torch.Tensor,
    *,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    test_rows: torch.Tensor | None = None,
    k: int = 3,
    seed: int = 42,
) -> list[StratifiedFold]:
    if int(k) <= 1:
        raise ValueError("k must be greater than 1")
    labels = labels.to(torch.long).cpu()
    candidates = torch.unique(torch.cat([train_rows.to(torch.long).cpu(), valid_rows.to(torch.long).cpu()])).to(torch.long)
    if test_rows is not None and test_rows.numel() > 0:
        test_set = set(int(v) for v in test_rows.to(torch.long).cpu().tolist())
        candidates = torch.tensor([int(v) for v in candidates.tolist() if int(v) not in test_set], dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    by_fold: list[list[int]] = [[] for _ in range(int(k))]
    for cls in sorted(int(v) for v in torch.unique(labels[candidates]).tolist()):
        cls_rows = candidates[labels[candidates] == cls]
        if cls_rows.numel() == 0:
            continue
        cls_rows = cls_rows[torch.randperm(cls_rows.numel(), generator=generator)]
        for idx, row in enumerate(cls_rows.tolist()):
            by_fold[idx % int(k)].append(int(row))
    all_rows = set(int(v) for v in candidates.tolist())
    folds: list[StratifiedFold] = []
    for fold_rows in by_fold:
        valid = torch.tensor(sorted(fold_rows), dtype=torch.long)
        train = torch.tensor(sorted(all_rows - set(fold_rows)), dtype=torch.long)
        folds.append(StratifiedFold(train_rows=train, valid_rows=valid))
    return folds
