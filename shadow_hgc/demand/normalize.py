from __future__ import annotations

import torch


def destination_row_normalize(
    edge_index: torch.Tensor,
    num_dst_nodes: int,
    raw_edge_weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute alpha_uv = w_uv / sum_{v' in N(u)} w_uv' by destination row."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    num_edges = edge_index.shape[1]
    if raw_edge_weight is None:
        raw_edge_weight = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
    else:
        raw_edge_weight = raw_edge_weight.to(device=edge_index.device, dtype=torch.float32)
    if raw_edge_weight.shape != (num_edges,):
        raise ValueError("raw_edge_weight must have shape [num_edges]")

    dst = edge_index[1]
    denom = torch.zeros(num_dst_nodes, dtype=raw_edge_weight.dtype, device=edge_index.device)
    denom.index_add_(0, dst, raw_edge_weight)
    return raw_edge_weight / denom[dst].clamp_min(eps)


def destination_degrees(edge_index: torch.Tensor, num_dst_nodes: int) -> torch.Tensor:
    """Count incoming edges under the fixed source-to-destination convention."""

    counts = torch.zeros(num_dst_nodes, dtype=torch.long, device=edge_index.device)
    if edge_index.numel() == 0:
        return counts
    ones = torch.ones(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    counts.index_add_(0, edge_index[1], ones)
    return counts
