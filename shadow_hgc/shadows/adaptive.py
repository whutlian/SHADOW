from __future__ import annotations

import math


def adaptive_shadow_budgets(
    rank_by_relation: dict[str, dict[str, float]],
    *,
    effective_M_tau: int,
    shadow_min_per_relation: int = 8,
    shadow_max_multiplier: float = 2.0,
    beta: float = 1.0,
) -> dict[str, int]:
    """Allocate relation shadow capacity from train-target rank diagnostics only."""

    max_budget = max(shadow_min_per_relation, int(math.floor(float(shadow_max_multiplier) * int(effective_M_tau))))
    budgets: dict[str, int] = {}
    for relation, diag in sorted(rank_by_relation.items()):
        rank_score = max(float(diag.get("stable_rank", 0.0)), float(diag.get("entropy_effective_rank", 0.0)))
        raw = int(math.ceil(float(beta) * rank_score * math.log1p(max(1, int(effective_M_tau)))))
        budgets[relation] = int(min(max(raw, int(shadow_min_per_relation)), max_budget))
    return budgets


def adaptive_assignment_b(reconstruction_error: float, *, b_max: int = 4) -> int:
    if reconstruction_error < 0.35:
        value = 1
    elif reconstruction_error < 0.60:
        value = 2
    else:
        value = 4
    return int(max(1, min(value, int(b_max))))
