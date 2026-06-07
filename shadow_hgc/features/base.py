from __future__ import annotations

import torch


def featureless_source_neighbor_mean(
    *,
    source_to_target_edge_index: torch.Tensor,
    target_base_features: torch.Tensor,
    num_source_nodes: int,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Leakage-safe featureless source initializer from neighboring target base features."""

    if fallback is None:
        fallback = target_base_features.mean(dim=0)
    src, dst = source_to_target_edge_index[0], source_to_target_edge_index[1]
    out = torch.zeros(num_source_nodes, target_base_features.shape[1], dtype=target_base_features.dtype)
    deg = torch.zeros(num_source_nodes, dtype=target_base_features.dtype)
    out.index_add_(0, src, target_base_features[dst])
    deg.index_add_(0, src, torch.ones_like(src, dtype=target_base_features.dtype))
    mask = deg > 0
    out[mask] = out[mask] / deg[mask].unsqueeze(-1)
    out[~mask] = fallback.to(out.dtype)
    return out
