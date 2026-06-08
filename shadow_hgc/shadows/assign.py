from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass
class TopBAssignmentResult:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    reconstruction: torch.Tensor
    topk_index: torch.Tensor
    topk_weight: torch.Tensor


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


def _solve_nonnegative_ridge(y: torch.Tensor, z: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    """Small exact active-set search for b<=4 nonnegative ridge regression."""

    b = z.shape[0]
    best_weight = torch.zeros(b, dtype=y.dtype, device=y.device)
    best_obj = torch.sum(y * y)
    if b == 0:
        return best_weight
    eye = torch.eye(b, dtype=y.dtype, device=y.device)
    for size in range(1, b + 1):
        for subset in combinations(range(b), size):
            idx = torch.tensor(subset, dtype=torch.long, device=y.device)
            zs = z[idx]
            gram = zs @ zs.T + float(ridge_lambda) * eye[:size, :size]
            rhs = zs @ y
            try:
                coeff = torch.linalg.solve(gram, rhs)
            except RuntimeError:
                coeff = torch.linalg.pinv(gram) @ rhs
            if bool(torch.any(coeff < -1e-8).item()):
                continue
            coeff = coeff.clamp_min(0.0)
            recon = coeff.unsqueeze(0) @ zs
            obj = torch.sum((y - recon.squeeze(0)) ** 2) + float(ridge_lambda) * torch.sum(coeff * coeff)
            if float(obj.item()) < float(best_obj.item()):
                candidate = torch.zeros(b, dtype=y.dtype, device=y.device)
                candidate[idx] = coeff
                best_weight = candidate
                best_obj = obj
    return best_weight


def topb_nonnegative_assignment(
    demand: torch.Tensor,
    shadow_features: torch.Tensor,
    *,
    b: int,
    ridge_lambda: float = 1e-4,
) -> TopBAssignmentResult:
    """Assign each demand row to up to b nearest shadows with nonnegative weights."""

    if demand.ndim != 2 or shadow_features.ndim != 2:
        raise ValueError("demand and shadow_features must be matrices")
    if shadow_features.shape[0] == 0:
        raise ValueError("cannot assign to an empty shadow pool")
    b = max(1, min(int(b), int(shadow_features.shape[0]), 4))
    dist = torch.cdist(demand, shadow_features)
    _, topk_index = torch.topk(dist, k=b, largest=False, dim=1)
    topk_weight = torch.zeros(demand.shape[0], b, dtype=demand.dtype, device=demand.device)
    reconstruction = torch.zeros_like(demand)
    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []
    edge_weight: list[torch.Tensor] = []
    for row in range(demand.shape[0]):
        idx = topk_index[row]
        z = shadow_features[idx].to(demand.device, demand.dtype)
        weights = _solve_nonnegative_ridge(demand[row], z, ridge_lambda)
        topk_weight[row] = weights
        reconstruction[row] = weights.unsqueeze(0) @ z
        positive = weights > 1e-10
        if not bool(positive.any()):
            nearest = idx[:1]
            edge_src.append(nearest)
            edge_dst.append(torch.tensor([row], dtype=torch.long, device=demand.device))
            edge_weight.append(torch.ones(1, dtype=demand.dtype, device=demand.device))
            reconstruction[row] = shadow_features[nearest[0]].to(demand.device, demand.dtype)
            topk_weight[row, 0] = 1.0
            continue
        kept = idx[positive]
        edge_src.append(kept)
        edge_dst.append(torch.full((kept.numel(),), row, dtype=torch.long, device=demand.device))
        edge_weight.append(weights[positive])
    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    return TopBAssignmentResult(
        edge_index=edge_index.to(torch.long),
        edge_weight=torch.cat(edge_weight).to(torch.float32),
        reconstruction=reconstruction,
        topk_index=topk_index,
        topk_weight=topk_weight,
    )
