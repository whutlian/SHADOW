from __future__ import annotations

import torch


def assign_nearest_shadow(demand: torch.Tensor, shadow_features: torch.Tensor) -> torch.Tensor:
    """Main b=1 nearest shadow assignment."""

    if shadow_features.shape[0] == 0:
        raise ValueError("cannot assign to an empty shadow pool")
    dist = torch.cdist(demand, shadow_features)
    return torch.argmin(dist, dim=1).to(torch.long)


def build_b1_shadow_edges(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build s_{r,pi(i)} -> p_i edges with non-negative unit weights."""

    dst = torch.arange(assignment.numel(), dtype=torch.long, device=assignment.device)
    edge_index = torch.stack([assignment.to(torch.long), dst], dim=0)
    edge_weight = torch.ones(assignment.numel(), dtype=torch.float32, device=assignment.device)
    return edge_index, edge_weight
