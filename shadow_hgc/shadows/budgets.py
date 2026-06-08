from __future__ import annotations

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.prototype.budgets import allocate_shadow_budgets


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

    by_name = allocate_shadow_budgets(
        effective_target_prototypes=effective_M_tau,
        relations=relations,
        shadow_ratio_target_target=target_target_ratio,
        shadow_ratio_non_target=non_target_ratio,
        min_shadow_per_relation=min_shadows_per_relation,
        target_type=target_type,
    )
    return {relation: by_name[str(relation)] for relation in relations}
