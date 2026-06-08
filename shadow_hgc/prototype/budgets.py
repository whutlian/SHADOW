from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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


def _round_budget(value: float, rounding: str) -> int:
    if rounding == "nearest":
        return int(round(value))
    if rounding == "ceil":
        return int(math.ceil(value))
    if rounding == "floor":
        return int(math.floor(value))
    raise ValueError("rounding must be one of: nearest, ceil, floor")


def compute_target_budget_from_ratio(
    *,
    num_train_target_nodes: int,
    num_train_classes: int,
    ratio: float,
    min_proto_per_class: int,
    max_target_budget: int | None = None,
    rounding: str = "nearest",
) -> dict:
    if num_train_target_nodes <= 0:
        raise ValueError("num_train_target_nodes must be positive")
    if num_train_classes <= 0:
        raise ValueError("num_train_classes must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if min_proto_per_class <= 0:
        raise ValueError("min_proto_per_class must be positive")
    requested = max(1, _round_budget(float(ratio) * int(num_train_target_nodes), rounding))
    if max_target_budget is not None:
        if max_target_budget <= 0:
            raise ValueError("max_target_budget must be positive when provided")
        requested = min(requested, int(max_target_budget))
    min_required = int(num_train_classes) * int(min_proto_per_class)
    effective = max(requested, min_required)
    return {
        "budget_mode": "ratio",
        "ratio": float(ratio),
        "ratio_base": "train_target",
        "num_train_target_nodes": int(num_train_target_nodes),
        "num_train_classes": int(num_train_classes),
        "min_proto_per_class": int(min_proto_per_class),
        "requested_target_budget": int(requested),
        "min_required_target_budget": int(min_required),
        "effective_target_prototypes": int(effective),
        "effective_target_ratio": float(effective / num_train_target_nodes),
        "budget_rounding": rounding,
        "max_target_budget": None if max_target_budget is None else int(max_target_budget),
        "budget_upshifted": bool(effective > requested),
    }


def allocate_classwise_budgets(
    *,
    class_counts: Mapping[int, int],
    target_budget: int,
    min_proto_per_class: int,
    exponent: float = 0.5,
) -> dict[int, int]:
    if target_budget <= 0:
        raise ValueError("target_budget must be positive")
    if min_proto_per_class <= 0:
        raise ValueError("min_proto_per_class must be positive")
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    active_counts = {int(cls): int(count) for cls, count in class_counts.items() if int(count) > 0}
    if not active_counts:
        raise ValueError("class_counts must include at least one class with training examples")
    min_required = len(active_counts) * int(min_proto_per_class)
    target_total = max(int(target_budget), min_required)

    weights = {cls: float(count) ** float(exponent) for cls, count in active_counts.items()}
    weight_sum = sum(weights.values())
    raw = {cls: target_total * weight / weight_sum for cls, weight in weights.items()}
    floors = {cls: int(math.floor(value)) for cls, value in raw.items()}
    budgets = {cls: max(int(min_proto_per_class), floors[cls]) for cls in active_counts}

    def total() -> int:
        return sum(budgets.values())

    while total() < target_total:
        candidates = sorted(
            active_counts,
            key=lambda cls: (raw[cls] - math.floor(raw[cls]), raw[cls], active_counts[cls], -cls),
            reverse=True,
        )
        for cls in candidates:
            if total() >= target_total:
                break
            budgets[cls] += 1

    while total() > target_total:
        candidates = sorted(
            [cls for cls in active_counts if budgets[cls] > min_proto_per_class],
            key=lambda cls: (raw[cls] - math.floor(raw[cls]), raw[cls], active_counts[cls], -cls),
        )
        if not candidates:
            break
        for cls in candidates:
            if total() <= target_total:
                break
            if budgets[cls] > min_proto_per_class:
                budgets[cls] -= 1
    return budgets


def allocate_shadow_budgets(
    *,
    effective_target_prototypes: int,
    relations: Iterable,
    shadow_ratio_target_target: float,
    shadow_ratio_non_target: float,
    min_shadow_per_relation: int,
    target_type: str | None = None,
) -> dict[str, int]:
    if effective_target_prototypes <= 0:
        raise ValueError("effective_target_prototypes must be positive")
    if shadow_ratio_target_target < 0 or shadow_ratio_non_target < 0:
        raise ValueError("shadow ratios must be non-negative")
    if min_shadow_per_relation <= 0:
        raise ValueError("min_shadow_per_relation must be positive")
    relation_list = list(relations)
    if target_type is None and relation_list:
        destinations = {getattr(rel, "destination_type", None) for rel in relation_list}
        if len(destinations) == 1:
            target_type = next(iter(destinations))
    target_target = [
        rel for rel in relation_list
        if getattr(rel, "source_type", None) == target_type and getattr(rel, "destination_type", None) == target_type
    ]
    non_target = [rel for rel in relation_list if rel not in target_target]
    budgets: dict[str, int] = {}
    if target_target:
        value = max(
            int(min_shadow_per_relation),
            int(math.ceil(float(shadow_ratio_target_target) * effective_target_prototypes / len(target_target))),
        )
        budgets.update({str(rel): value for rel in target_target})
    if non_target:
        value = max(
            int(min_shadow_per_relation),
            int(math.ceil(float(shadow_ratio_non_target) * effective_target_prototypes / len(non_target))),
        )
        budgets.update({str(rel): value for rel in non_target})
    return budgets


def validate_budget_mode_args(
    *,
    budget_mode: str,
    ratio: float | None,
    target_budget: int | None,
) -> None:
    if budget_mode not in {"ratio", "count"}:
        raise ValueError("budget_mode must be either ratio or count")
    if ratio is not None and target_budget is not None:
        raise ValueError("provide either ratio or target_budget, not both")
    if budget_mode == "ratio" and ratio is None:
        raise ValueError("ratio budget mode requires ratio")
    if budget_mode == "count" and target_budget is None:
        raise ValueError("count budget mode requires target_budget")


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
    class_counts = {int(c.item()): int(count.item()) for c, count in zip(classes, counts)}
    budgets = allocate_classwise_budgets(
        class_counts=class_counts,
        target_budget=target_total,
        min_proto_per_class=min_proto_per_class,
        exponent=budget_alpha,
    )
    return BudgetResult(
        budgets=budgets,
        requested_M_tau=int(M_tau),
        effective_M_tau=int(sum(budgets.values())),
        num_classes=num_classes,
        min_proto_per_class=int(min_proto_per_class),
        budget_alpha=float(budget_alpha),
        budget_upshifted=target_total > M_tau,
    )
