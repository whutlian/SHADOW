from __future__ import annotations

import math

import torch


def class_wise_budget(labels: torch.Tensor, train_idx: torch.Tensor, M_tau: int) -> dict[int, int]:
    train_labels = labels[train_idx]
    classes, counts = torch.unique(train_labels, return_counts=True)
    roots = torch.sqrt(counts.to(torch.float64))
    raw = M_tau * roots / roots.sum()
    budgets = {int(c.item()): max(1, int(round(v.item()))) for c, v in zip(classes, raw)}

    def total() -> int:
        return sum(budgets.values())

    while total() > M_tau and len(budgets) > 0:
        candidates = [c for c in budgets if budgets[c] > 1]
        if not candidates:
            break
        c = max(candidates, key=lambda cls: budgets[cls] - float(raw[(classes == cls).nonzero()[0]].item()))
        budgets[c] -= 1
    while total() < M_tau:
        c = max(
            [int(value.item()) for value in classes],
            key=lambda cls: float(raw[(classes == cls).nonzero()[0]].item()) - budgets[cls],
        )
        budgets[c] += 1
    return budgets
