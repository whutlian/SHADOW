from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation


def block_normalize(blocks: list[torch.Tensor], *, eps: float = 1e-12) -> torch.Tensor:
    normalized = []
    for block in blocks:
        denom = torch.linalg.norm(block, dim=1, keepdim=True).clamp_min(eps)
        normalized.append(block / denom)
    return torch.cat(normalized, dim=1)


def build_target_signature(
    psi_target: torch.Tensor,
    demand_by_relation: dict[DirectedRelation, torch.Tensor],
    degree_features: torch.Tensor,
    *,
    eta: float = 0.1,
    relation_order: list[DirectedRelation] | None = None,
    extra_blocks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if relation_order is None:
        relation_order = list(demand_by_relation)
    blocks = [psi_target] + [demand_by_relation[relation] for relation in relation_order] + [eta * degree_features]
    if extra_blocks:
        blocks.extend(extra_blocks)
    return block_normalize(blocks)
