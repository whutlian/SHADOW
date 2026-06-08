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


def _rank_score(diag: dict[str, float]) -> float:
    return max(float(diag.get("stable_rank", 0.0)), float(diag.get("entropy_effective_rank", 0.0)), 0.0)


def adaptive_shadow_budgets_global_cap(
    rank_by_relation: dict[str, dict[str, float]],
    *,
    effective_M_tau: int,
    total_shadow_budget: int,
    shadow_min_per_relation: int,
    beta: float = 1.0,
) -> dict[str, int]:
    """Allocate train-diagnostic shadow budgets under one hard global cap."""

    relations = sorted(rank_by_relation)
    total_shadow_budget = max(0, int(total_shadow_budget))
    shadow_min_per_relation = max(0, int(shadow_min_per_relation))
    if not relations:
        return {}
    if total_shadow_budget == 0:
        return {relation: 0 for relation in relations}

    raw_scores = {relation: _rank_score(rank_by_relation[relation]) for relation in relations}
    if all(score == 0.0 for score in raw_scores.values()):
        scores = {relation: 1.0 for relation in relations}
    else:
        scores = {
            relation: (score if score > 0.0 else 0.0) ** max(float(beta), 0.0)
            for relation, score in raw_scores.items()
        }
        if sum(scores.values()) <= 0.0:
            scores = {relation: 1.0 for relation in relations}

    budgets = {relation: 0 for relation in relations}
    min_total = shadow_min_per_relation * len(relations)
    if shadow_min_per_relation > 0 and total_shadow_budget < min_total:
        remaining = total_shadow_budget
        for relation in sorted(relations, key=lambda rel: (-scores[rel], rel)):
            if remaining <= 0:
                break
            grant = min(shadow_min_per_relation, remaining)
            budgets[relation] = int(grant)
            remaining -= grant
        return budgets

    for relation in relations:
        budgets[relation] = shadow_min_per_relation
    remaining = total_shadow_budget - sum(budgets.values())
    if remaining <= 0:
        return budgets

    score_total = sum(scores.values())
    quotas = {relation: remaining * scores[relation] / score_total for relation in relations}
    for relation in relations:
        grant = int(math.floor(quotas[relation]))
        budgets[relation] += grant
        remaining -= grant

    remainders = sorted(
        relations,
        key=lambda relation: (-(quotas[relation] - math.floor(quotas[relation])), -scores[relation], relation),
    )
    for relation in remainders:
        if remaining <= 0:
            break
        budgets[relation] += 1
        remaining -= 1

    return budgets


def adaptive_assignment_b(reconstruction_error: float, *, b_max: int = 4) -> int:
    if reconstruction_error < 0.35:
        value = 1
    elif reconstruction_error < 0.60:
        value = 2
    else:
        value = 4
    return int(max(1, min(value, int(b_max))))
