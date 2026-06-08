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
    mean_adaptive_k: float = 0.0
    median_adaptive_k: float = 0.0
    max_adaptive_k: int = 0


@dataclass
class SparseTopKTransition:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    skeleton_message: torch.Tensor
    total_mass: float
    skeleton_mass: float
    k_by_row: list[int] | None = None


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


def sparse_topk_transition_message(
    *,
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    target_to_cell: torch.Tensor,
    cell_sizes: torch.Tensor,
    num_cells: int,
    prototype_features: torch.Tensor,
    k_s: int,
    skeleton_policy: str = "fixed_k",
    skeleton_coverage: float = 0.65,
    skeleton_k_max: int = 8,
) -> SparseTopKTransition:
    """Compute top-k transition messages without materializing dense cell-cell matrices."""

    empty_edges = torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
    empty_weight = torch.empty(0, dtype=alpha.dtype, device=alpha.device)
    skeleton_message = torch.zeros(
        num_cells,
        prototype_features.shape[1],
        dtype=prototype_features.dtype,
        device=prototype_features.device,
    )
    if edge_index.numel() == 0 or num_cells == 0:
        return SparseTopKTransition(empty_edges, empty_weight, skeleton_message, 0.0, 0.0, [])

    src_cell = target_to_cell[edge_index[0]]
    dst_cell = target_to_cell[edge_index[1]]
    valid = (src_cell >= 0) & (dst_cell >= 0)
    if not bool(valid.any()):
        return SparseTopKTransition(empty_edges, empty_weight, skeleton_message, 0.0, 0.0, [])

    src_cell = src_cell[valid].to(torch.long)
    dst_cell = dst_cell[valid].to(torch.long)
    scaled_mass = alpha[valid] / cell_sizes.to(alpha.device, alpha.dtype)[dst_cell]
    flat = dst_cell * int(num_cells) + src_cell
    unique_flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    pair_mass = torch.zeros(unique_flat.shape[0], dtype=scaled_mass.dtype, device=scaled_mass.device)
    pair_mass.index_add_(0, inverse, scaled_mass)

    total_mass = float(pair_mass.sum().item())
    if k_s <= 0 or pair_mass.numel() == 0:
        return SparseTopKTransition(empty_edges, empty_weight, skeleton_message, total_mass, 0.0, [0] * int(num_cells))

    pair_dst = torch.div(unique_flat, int(num_cells), rounding_mode="floor").to(torch.long)
    pair_src = (unique_flat % int(num_cells)).to(torch.long)
    counts = torch.bincount(pair_dst, minlength=num_cells)
    offsets = torch.cumsum(counts, dim=0) - counts
    active_dst = torch.nonzero(counts > 0, as_tuple=False).flatten()

    edge_pieces: list[torch.Tensor] = []
    weight_pieces: list[torch.Tensor] = []
    skeleton_mass = 0.0
    k_by_row = [0] * int(num_cells)
    for dst in active_dst.tolist():
        start = int(offsets[dst].item())
        end = start + int(counts[dst].item())
        masses = pair_mass[start:end]
        sources = pair_src[start:end]
        if skeleton_policy == "coverage":
            k = min(int(skeleton_k_max), int(masses.numel()))
        elif skeleton_policy == "fixed_k":
            k = min(int(k_s), int(masses.numel()))
        else:
            raise ValueError("skeleton_policy must be fixed_k or coverage")
        if k <= 0:
            continue
        values, indices = torch.topk(masses, k=k)
        if skeleton_policy == "coverage":
            row_total = float(masses.sum().item())
            if row_total > 0.0:
                cumulative = torch.cumsum(values, dim=0)
                reached = torch.nonzero(cumulative >= float(skeleton_coverage) * row_total, as_tuple=False)
                if reached.numel() > 0:
                    keep = int(reached[0].item()) + 1
                    values = values[:keep]
                    indices = indices[:keep]
        positive = values > 0
        if not bool(positive.any()):
            continue
        values = values[positive]
        sources = sources[indices[positive]]
        k_by_row[int(dst)] = int(values.numel())
        dst_values = torch.full_like(sources, dst)
        edge_pieces.append(torch.stack([sources, dst_values], dim=0))
        weight_pieces.append(values)
        skeleton_message[dst] = (
            values.to(device=prototype_features.device, dtype=prototype_features.dtype).unsqueeze(0)
            @ prototype_features[sources.to(prototype_features.device)]
        ).squeeze(0)
        skeleton_mass += float(values.sum().item())

    if not edge_pieces:
        return SparseTopKTransition(empty_edges, empty_weight, skeleton_message, total_mass, 0.0, k_by_row)
    return SparseTopKTransition(
        edge_index=torch.cat(edge_pieces, dim=1),
        edge_weight=torch.cat(weight_pieces, dim=0),
        skeleton_message=skeleton_message,
        total_mass=total_mass,
        skeleton_mass=skeleton_mass,
        k_by_row=k_by_row,
    )


def compute_target_target_residual_skeleton(
    *,
    demand: torch.Tensor,
    prototype_features: torch.Tensor,
    target_to_cell: torch.Tensor,
    cell_members: list[torch.Tensor],
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    k_s: int,
    demand_row_by_target: torch.Tensor | None = None,
    skeleton_policy: str = "fixed_k",
    skeleton_coverage: float = 0.65,
    skeleton_k_max: int = 8,
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
            rows = members.to(demand.device)
            if demand_row_by_target is not None:
                rows = demand_row_by_target.to(demand.device)[rows]
                rows = rows[rows >= 0]
            if rows.numel() > 0:
                D[cell_id] = demand[rows].mean(dim=0)

    sparse_topk = sparse_topk_transition_message(
        edge_index=edge_index,
        alpha=alpha,
        target_to_cell=target_to_cell.to(edge_index.device),
        cell_sizes=cell_sizes.to(edge_index.device),
        num_cells=num_cells,
        prototype_features=prototype_features,
        k_s=k_s,
        skeleton_policy=skeleton_policy,
        skeleton_coverage=skeleton_coverage,
        skeleton_k_max=skeleton_k_max,
    )
    D_skel = sparse_topk.skeleton_message.to(demand.device)
    residual = D - D_skel
    coverage = float(sparse_topk.skeleton_mass / (sparse_topk.total_mass + eps))
    residual_energy = float(torch.linalg.norm(residual).item() / (torch.linalg.norm(D).item() + eps))
    k_values = sparse_topk.k_by_row or []
    k_tensor = torch.tensor(k_values, dtype=torch.float32) if k_values else torch.empty(0)
    return SkeletonResult(
        D=D,
        S=torch.empty(0, 0, dtype=alpha.dtype, device=demand.device),
        S_top=torch.empty(0, 0, dtype=alpha.dtype, device=demand.device),
        residual=residual,
        skeleton_edge_index=sparse_topk.edge_index.to(demand.device),
        skeleton_edge_weight=sparse_topk.edge_weight.to(demand.device),
        skeleton_mass_coverage=coverage,
        residual_energy=residual_energy,
        mean_adaptive_k=float(k_tensor.mean().item()) if k_tensor.numel() else 0.0,
        median_adaptive_k=float(torch.median(k_tensor).item()) if k_tensor.numel() else 0.0,
        max_adaptive_k=int(k_tensor.max().item()) if k_tensor.numel() else 0,
    )
