from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CoverageSkeletonResult:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    S_top: torch.Tensor
    k_by_row: list[int]
    mean_k: float
    median_k: float
    max_k: int
    actual_coverage: float


def coverage_topk_skeleton(S: torch.Tensor, *, coverage: float = 0.65, k_max: int = 8) -> CoverageSkeletonResult:
    """Keep minimum top-k mass per row to reach coverage, without renormalization."""

    num_dst, num_src = S.shape
    S_top = torch.zeros_like(S)
    edge_pieces: list[torch.Tensor] = []
    weight_pieces: list[torch.Tensor] = []
    k_by_row: list[int] = []
    for dst in range(num_dst):
        row = S[dst]
        positive_count = int((row > 0).sum().item())
        if positive_count == 0 or k_max <= 0:
            k_by_row.append(0)
            continue
        k_limit = min(int(k_max), positive_count, num_src)
        values, indices = torch.topk(row, k=k_limit)
        row_total = float(row.sum().item())
        keep = k_limit
        if row_total > 0.0:
            cumulative = torch.cumsum(values, dim=0)
            reached = torch.nonzero(cumulative >= float(coverage) * row_total, as_tuple=False)
            if reached.numel() > 0:
                keep = int(reached[0].item()) + 1
        kept_values = values[:keep]
        kept_indices = indices[:keep]
        positive = kept_values > 0
        kept_values = kept_values[positive]
        kept_indices = kept_indices[positive]
        k_by_row.append(int(kept_values.numel()))
        if kept_values.numel() == 0:
            continue
        S_top[dst, kept_indices] = kept_values
        dst_values = torch.full_like(kept_indices, dst)
        edge_pieces.append(torch.stack([kept_indices, dst_values], dim=0))
        weight_pieces.append(kept_values)
    if edge_pieces:
        edge_index = torch.cat(edge_pieces, dim=1).to(torch.long)
        edge_weight = torch.cat(weight_pieces, dim=0)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=S.device)
        edge_weight = torch.empty(0, dtype=S.dtype, device=S.device)
    k_tensor = torch.tensor(k_by_row, dtype=torch.float32)
    actual = float(S_top.sum().item() / (S.sum().item() + 1e-12)) if S.numel() else 0.0
    return CoverageSkeletonResult(
        edge_index=edge_index,
        edge_weight=edge_weight,
        S_top=S_top,
        k_by_row=k_by_row,
        mean_k=float(k_tensor.mean().item()) if k_by_row else 0.0,
        median_k=float(torch.median(k_tensor).item()) if k_by_row else 0.0,
        max_k=max(k_by_row) if k_by_row else 0,
        actual_coverage=actual,
    )
