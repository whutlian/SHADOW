from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SkeletonResult:
    D: torch.Tensor
    S: torch.Tensor
    S_top: torch.Tensor
    residual: torch.Tensor
    skeleton_edge_index: torch.Tensor
    skeleton_edge_weight: torch.Tensor
    skeleton_mass_coverage: float
    residual_energy: float


def compute_transition_mass(
    *,
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    target_to_cell: torch.Tensor,
    cell_sizes: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """Compute S_ij with i as destination cell and j as source cell."""

    src_cell = target_to_cell[edge_index[0]]
    dst_cell = target_to_cell[edge_index[1]]
    valid = (src_cell >= 0) & (dst_cell >= 0)
    S = torch.zeros(num_cells, num_cells, dtype=alpha.dtype, device=alpha.device)
    if not bool(valid.any()):
        return S

    scale = alpha[valid] / cell_sizes.to(alpha.device, alpha.dtype)[dst_cell[valid]]
    flat = dst_cell[valid] * num_cells + src_cell[valid]
    S.view(-1).index_add_(0, flat, scale)
    return S


def topk_skeleton(S: torch.Tensor, k_s: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep top-k source cells per destination row without renormalizing weights."""

    num_dst, num_src = S.shape
    S_top = torch.zeros_like(S)
    if k_s <= 0 or num_dst == 0 or num_src == 0:
        empty_edges = torch.empty(2, 0, dtype=torch.long, device=S.device)
        empty_weight = torch.empty(0, dtype=S.dtype, device=S.device)
        return empty_edges, empty_weight, S_top

    k = min(k_s, num_src)
    values, indices = torch.topk(S, k=k, dim=1)
    row = torch.arange(num_dst, device=S.device).unsqueeze(1).expand_as(indices)
    positive = values > 0
    S_top[row[positive], indices[positive]] = values[positive]
    edge_index = torch.stack([indices[positive], row[positive]], dim=0).to(torch.long)
    edge_weight = values[positive]
    return edge_index, edge_weight, S_top


def compute_target_target_residual_skeleton(
    *,
    demand: torch.Tensor,
    prototype_features: torch.Tensor,
    target_to_cell: torch.Tensor,
    cell_members: list[torch.Tensor],
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    k_s: int,
    eps: float = 1e-12,
) -> SkeletonResult:
    """Decompose target-target demand into sparse skeleton plus residual demand."""

    num_cells = len(cell_members)
    cell_sizes = torch.tensor(
        [len(members) for members in cell_members],
        dtype=demand.dtype,
        device=demand.device,
    ).clamp_min(1)
    D = torch.zeros(num_cells, demand.shape[1], dtype=demand.dtype, device=demand.device)
    for cell_id, members in enumerate(cell_members):
        if len(members) > 0:
            D[cell_id] = demand[members.to(demand.device)].mean(dim=0)

    S = compute_transition_mass(
        edge_index=edge_index,
        alpha=alpha,
        target_to_cell=target_to_cell.to(edge_index.device),
        cell_sizes=cell_sizes.to(edge_index.device),
        num_cells=num_cells,
    ).to(demand.device)
    skeleton_edge_index, skeleton_edge_weight, S_top = topk_skeleton(S, k_s)
    D_skel = S_top @ prototype_features
    residual = D - D_skel
    coverage = float(S_top.sum().item() / (S.sum().item() + eps))
    residual_energy = float(torch.linalg.norm(residual).item() / (torch.linalg.norm(D).item() + eps))
    return SkeletonResult(
        D=D,
        S=S,
        S_top=S_top,
        residual=residual,
        skeleton_edge_index=skeleton_edge_index,
        skeleton_edge_weight=skeleton_edge_weight,
        skeleton_mass_coverage=coverage,
        residual_energy=residual_energy,
    )
