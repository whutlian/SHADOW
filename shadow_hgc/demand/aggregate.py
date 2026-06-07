from __future__ import annotations

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


def weighted_scatter_add(
    messages: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    num_dst_nodes: int,
) -> torch.Tensor:
    """Aggregate edge messages with explicit non-hidden edge weights."""

    if messages.ndim != 2:
        raise ValueError("messages must have shape [num_edges, feature_dim]")
    if dst.shape != (messages.shape[0],):
        raise ValueError("dst must have one destination index per message")
    if edge_weight.shape != (messages.shape[0],):
        raise ValueError("edge_weight must have one scalar per message")
    weighted = messages * edge_weight.to(messages.dtype).unsqueeze(-1)
    out = torch.zeros(
        num_dst_nodes,
        messages.shape[1],
        dtype=messages.dtype,
        device=messages.device,
    )
    out.index_add_(0, dst, weighted)
    return out


def aggregate_relation_demand(
    *,
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_dst_nodes: int,
    raw_edge_weight: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    edge_chunk_size: int | None = None,
    return_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Aggregate mu_u^r from actual source model features phi_source(v)."""

    if alpha is None:
        alpha = destination_row_normalize(edge_index, num_dst_nodes, raw_edge_weight)
    else:
        alpha = alpha.to(device=source_features.device, dtype=source_features.dtype)
    src, dst = edge_index[0], edge_index[1]
    if edge_chunk_size is None:
        demand = weighted_scatter_add(
            source_features[src],
            dst,
            alpha,
            num_dst_nodes=num_dst_nodes,
        )
    else:
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        demand = torch.zeros(
            num_dst_nodes,
            source_features.shape[1],
            dtype=source_features.dtype,
            device=source_features.device,
        )
        for start in range(0, edge_index.shape[1], edge_chunk_size):
            end = min(start + edge_chunk_size, edge_index.shape[1])
            chunk_src = src[start:end]
            chunk_dst = dst[start:end]
            chunk_weight = alpha[start:end]
            chunk_messages = source_features[chunk_src] * chunk_weight.to(source_features.dtype).unsqueeze(-1)
            demand.index_add_(0, chunk_dst, chunk_messages)
    if return_alpha:
        return demand, alpha
    return demand
