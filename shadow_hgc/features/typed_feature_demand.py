from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.features.projection import fixed_random_projection


@dataclass(frozen=True)
class TypedFeatureDemandResult:
    block: torch.Tensor
    diagnostics: dict


def _target_lookup(num_target_nodes: int, target_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    lookup = torch.full((int(num_target_nodes),), -1, dtype=torch.long, device=device)
    rows = target_rows.to(device=device, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long, device=device)
    return lookup


def compute_typed_feature_demand(
    *,
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_target_nodes: int,
    target_rows: torch.Tensor,
    chunk_size: int = 65536,
    projection_dim: int | None = None,
    projection_seed: int = 42,
    cache_all_nodes: bool = False,
    debug_allow_all_node_cache: bool = False,
) -> TypedFeatureDemandResult:
    """Compute one-hop typed feature demand using destination-row alpha.

    The function only materializes per-chunk `chunk_edges x feature_dim` messages,
    never the full `E x d` message matrix.
    """

    if cache_all_nodes and not debug_allow_all_node_cache:
        raise ValueError("all-node high-dimensional demand cache is forbidden outside debug mode")
    started = time.perf_counter()
    if source_features.ndim != 2:
        raise ValueError("source_features must have shape [num_source_nodes, feature_dim]")
    device = source_features.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    rows = target_rows.to(device=device, dtype=torch.long)
    x = source_features.to(torch.float32)
    if projection_dim is not None and int(projection_dim) < int(x.shape[1]):
        x = fixed_random_projection(x, out_dim=int(projection_dim), seed=int(projection_seed)).to(torch.float32)
    out = torch.zeros(rows.numel(), x.shape[1], dtype=torch.float32, device=device)
    if edge_index.numel() == 0 or rows.numel() == 0:
        return TypedFeatureDemandResult(
            block=out,
            diagnostics={
                "normalization": "destination_row",
                "feature_demand_dim": int(x.shape[1]),
                "feature_demand_cache_bytes": int(out.numel() * out.element_size()),
                "feature_demand_full_edge_scans": 1,
                "full_edge_scans": 1,
                "materialized_full_e_by_d": False,
                "max_edge_chunk_size": 0,
                "feature_demand_precompute_time_s": float(time.perf_counter() - started),
            },
        )
    alpha = destination_row_normalize(edge_index, int(num_target_nodes)).to(device=device, dtype=torch.float32)
    lookup = _target_lookup(num_target_nodes, rows, device)
    max_chunk = 0
    for start in range(0, int(edge_index.shape[1]), int(chunk_size)):
        end = min(int(edge_index.shape[1]), start + int(chunk_size))
        max_chunk = max(max_chunk, end - start)
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]
        local_dst = lookup[dst]
        mask = local_dst >= 0
        if bool(mask.any()):
            messages = x[src[mask]] * alpha[start:end][mask].unsqueeze(1)
            out.index_add_(0, local_dst[mask], messages)
    diagnostics = {
        "normalization": "destination_row",
        "feature_demand_dim": int(x.shape[1]),
        "feature_demand_cache_bytes": int(out.numel() * out.element_size()),
        "feature_demand_full_edge_scans": 1,
        "feature_demand_precompute_time_s": float(time.perf_counter() - started),
        "full_edge_scans": 1,
        "target_rows_only": True,
        "materialized_full_e_by_d": False,
        "max_edge_chunk_size": int(max_chunk),
        "cache_all_nodes": bool(cache_all_nodes),
    }
    return TypedFeatureDemandResult(block=out, diagnostics=diagnostics)


def estimate_typed_feature_demand_cache_bytes(*, num_target_rows: int, feature_dim: int, dtype_bytes: int = 4) -> int:
    return int(num_target_rows) * int(feature_dim) * int(dtype_bytes)
