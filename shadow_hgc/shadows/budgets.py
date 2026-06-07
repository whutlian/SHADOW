from __future__ import annotations

import math

from shadow_hgc.data.schemas import DirectedRelation


def resolve_shadow_budgets(
    *,
    relations: list[DirectedRelation],
    target_type: str,
    effective_M_tau: int,
    requested_M_r: int | dict | None = None,
    non_target_ratio: float = 1.0,
    target_target_ratio: float = 0.5,
    min_shadows_per_relation: int = 8,
) -> dict[DirectedRelation, int]:
    if requested_M_r is not None:
        if isinstance(requested_M_r, dict):
            return {
                relation: int(requested_M_r.get(str(relation), requested_M_r.get(relation, min_shadows_per_relation)))
                for relation in relations
            }
        return {relation: int(requested_M_r) for relation in relations}

    target_target = [relation for relation in relations if relation.source_type == target_type]
    non_target = [relation for relation in relations if relation.source_type != target_type]
    budgets: dict[DirectedRelation, int] = {}
    if target_target:
        value = max(min_shadows_per_relation, math.ceil(target_target_ratio * effective_M_tau / len(target_target)))
        budgets.update({relation: int(value) for relation in target_target})
    if non_target:
        value = max(min_shadows_per_relation, math.ceil(non_target_ratio * effective_M_tau / len(non_target)))
        budgets.update({relation: int(value) for relation in non_target})
    return budgets
