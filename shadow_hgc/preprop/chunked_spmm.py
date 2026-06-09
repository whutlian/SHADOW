from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.preprop.normalization import destination_alpha


@dataclass(frozen=True)
class TrueChunkedSpMMResult:
    block: torch.Tensor
    diagnostics: dict[str, Any]


def chunked_destination_row_spmm(
    *,
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_dst_nodes: int,
    dst_rows: torch.Tensor | None = None,
    edge_chunk_size: int = 2_000_000,
) -> TrueChunkedSpMMResult:
    started = time.perf_counter()
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if source_features.ndim != 2:
        raise ValueError("source_features must have shape [num_source_nodes, dim]")
    edge_chunk_size = max(1, int(edge_chunk_size))
    num_dst_nodes = int(num_dst_nodes)
    if dst_rows is None:
        dst_rows = torch.arange(num_dst_nodes, dtype=torch.long)
    else:
        dst_rows = dst_rows.to(torch.long).cpu()
    x = source_features.detach().to(torch.float32).cpu()
    edges = edge_index.to(torch.long).cpu()
    out = torch.zeros(int(dst_rows.numel()), int(x.shape[1]), dtype=torch.float32)
    max_chunk = 0
    if edges.numel() > 0 and dst_rows.numel() > 0:
        alpha = destination_alpha(edges, num_dst_nodes=num_dst_nodes).cpu()
        lookup = torch.full((num_dst_nodes,), -1, dtype=torch.long)
        lookup[dst_rows] = torch.arange(dst_rows.numel(), dtype=torch.long)
        for start in range(0, int(edges.shape[1]), edge_chunk_size):
            end = min(int(edges.shape[1]), start + edge_chunk_size)
            max_chunk = max(max_chunk, end - start)
            src = edges[0, start:end]
            dst = edges[1, start:end]
            local_dst = lookup[dst]
            mask = local_dst >= 0
            if bool(mask.any()):
                messages = x[src[mask]] * alpha[start:end][mask].unsqueeze(1)
                out.index_add_(0, local_dst[mask], messages)
    return TrueChunkedSpMMResult(
        block=out,
        diagnostics={
            "normalization": "destination_row",
            "edge_scans": 1,
            "full_edge_scans": 1,
            "max_edge_chunk_size": int(max_chunk),
            "edge_chunk_size": int(edge_chunk_size),
            "uses_e_by_d_materialization": False,
            "materialized_full_e_by_d": False,
            "wall_time_s": float(time.perf_counter() - started),
        },
    )
