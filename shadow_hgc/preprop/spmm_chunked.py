from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.preprop.normalization import destination_alpha


@dataclass(frozen=True)
class ChunkedSpMMResult:
    block: torch.Tensor
    diagnostics: dict[str, Any]


def _target_lookup(num_dst_nodes: int, dst_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    lookup = torch.full((int(num_dst_nodes),), -1, dtype=torch.long, device=device)
    rows = dst_rows.to(device=device, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long, device=device)
    return lookup


def chunked_destination_spmm(
    *,
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_dst_nodes: int,
    dst_rows: torch.Tensor,
    edge_chunk_size: int = 2_000_000,
) -> ChunkedSpMMResult:
    started = time.perf_counter()
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if source_features.ndim != 2:
        raise ValueError("source_features must have shape [num_source_nodes, dim]")
    edge_chunk_size = max(1, int(edge_chunk_size))
    device = source_features.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    dst_rows = dst_rows.to(device=device, dtype=torch.long)
    x = source_features.to(torch.float32)
    out = torch.zeros(dst_rows.numel(), int(x.shape[1]), dtype=torch.float32, device=device)
    max_chunk = 0
    if edge_index.numel() > 0 and dst_rows.numel() > 0:
        alpha = destination_alpha(edge_index, num_dst_nodes=int(num_dst_nodes)).to(device=device, dtype=torch.float32)
        lookup = _target_lookup(int(num_dst_nodes), dst_rows, device)
        for start in range(0, int(edge_index.shape[1]), edge_chunk_size):
            end = min(int(edge_index.shape[1]), start + edge_chunk_size)
            max_chunk = max(max_chunk, end - start)
            src = edge_index[0, start:end]
            dst = edge_index[1, start:end]
            local_dst = lookup[dst]
            mask = local_dst >= 0
            if bool(mask.any()):
                messages = x[src[mask]] * alpha[start:end][mask].unsqueeze(1)
                out.index_add_(0, local_dst[mask], messages)
    return ChunkedSpMMResult(
        block=out,
        diagnostics={
            "normalization": "destination_row",
            "full_edge_scans": 1,
            "edge_scans": 1,
            "max_edge_chunk_size": int(max_chunk),
            "edge_chunk_size": int(edge_chunk_size),
            "uses_e_by_d_materialization": False,
            "materialized_full_e_by_d": False,
            "wall_time_s": float(time.perf_counter() - started),
        },
    )
