from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BudgetResult(Mapping[int, int]):
    budgets: dict[int, int]
    requested_M_tau: int
    effective_M_tau: int
    num_classes: int
    min_proto_per_class: int
    budget_alpha: float
    budget_upshifted: bool

    def __getitem__(self, key: int) -> int:
        return self.budgets[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self.budgets)

    def __len__(self) -> int:
        return len(self.budgets)

    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return self.budgets == other
        return super().__eq__(other)


def class_wise_budget(
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    M_tau: int,
    *,
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    strict: bool = False,
) -> BudgetResult:
    if M_tau <= 0:
        raise ValueError("M_tau must be positive")
    if min_proto_per_class <= 0:
        raise ValueError("min_proto_per_class must be positive")
    if budget_alpha < 0:
        raise ValueError("budget_alpha must be non-negative")
    train_labels = labels[train_idx]
    classes, counts = torch.unique(train_labels, return_counts=True)
    classes = classes[classes >= 0]
    counts = torch.stack([(train_labels == class_id).sum() for class_id in classes]).to(torch.long)
    num_classes = int(classes.numel())
    minimum_total = num_classes * min_proto_per_class
    if M_tau < minimum_total and strict:
        raise ValueError(
            f"requested M_tau={M_tau} is below num_classes * min_proto_per_class={minimum_total}"
        )
    target_total = max(M_tau, minimum_total)
    weights = counts.to(torch.float64).pow(budget_alpha)
    raw = target_total * weights / weights.sum()
    budgets = {
        int(c.item()): max(min_proto_per_class, int(round(v.item())))
        for c, v in zip(classes, raw)
    }

    def total() -> int:
        return sum(budgets.values())

    while total() > target_total and len(budgets) > 0:
        candidates = [c for c in budgets if budgets[c] > min_proto_per_class]
        if not candidates:
            break
        c = max(candidates, key=lambda cls: budgets[cls] - float(raw[(classes == cls).nonzero()[0]].item()))
        budgets[c] -= 1
    while total() < target_total:
        c = max(
            [int(value.item()) for value in classes],
            key=lambda cls: float(raw[(classes == cls).nonzero()[0]].item()) - budgets[cls],
        )
        budgets[c] += 1
    return BudgetResult(
        budgets=budgets,
        requested_M_tau=int(M_tau),
        effective_M_tau=int(total()),
        num_classes=num_classes,
        min_proto_per_class=int(min_proto_per_class),
        budget_alpha=float(budget_alpha),
        budget_upshifted=target_total > M_tau,
    )
