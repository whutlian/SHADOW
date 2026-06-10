from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class StreamingSpMMResult:
    block: torch.Tensor
    diagnostics: dict[str, Any]


def streaming_destination_row_spmm(
    *,
    edge_stream_factory: Callable[[], Iterable],
    source_feature_getter: Callable[[torch.Tensor], torch.Tensor],
    feature_dim: int,
    num_dst_nodes: int,
    dst_rows: torch.Tensor | None = None,
) -> StreamingSpMMResult:
    """Destination-row-normalized SpMM over a re-iterable edge stream.

    The edge stream yields chunks with `src`, `dst`, and non-negative `weight`.
    This function never materializes a stacked `[2, E]` edge_index or an `E x d`
    message matrix.
    """

    started = time.perf_counter()
    num_dst_nodes = int(num_dst_nodes)
    feature_dim = int(feature_dim)
    if dst_rows is None:
        dst_rows = torch.arange(num_dst_nodes, dtype=torch.long)
    else:
        dst_rows = dst_rows.to(torch.long).cpu()
    degree_weight_sum = torch.zeros(num_dst_nodes, dtype=torch.float32)
    max_chunk = 0
    num_edges = 0

    for chunk in edge_stream_factory():
        dst = chunk.dst.to(torch.long).cpu()
        weight = chunk.weight.to(torch.float32).cpu()
        if dst.numel() != weight.numel():
            raise ValueError("edge chunk dst and weight must have the same length")
        max_chunk = max(max_chunk, int(dst.numel()))
        num_edges += int(dst.numel())
        if dst.numel() > 0:
            degree_weight_sum.index_add_(0, dst, weight)

    lookup = torch.full((num_dst_nodes,), -1, dtype=torch.long)
    lookup[dst_rows] = torch.arange(dst_rows.numel(), dtype=torch.long)
    out = torch.zeros(int(dst_rows.numel()), feature_dim, dtype=torch.float32)

    for chunk in edge_stream_factory():
        src = chunk.src.to(torch.long).cpu()
        dst = chunk.dst.to(torch.long).cpu()
        weight = chunk.weight.to(torch.float32).cpu()
        local_dst = lookup[dst]
        mask = local_dst >= 0
        if not bool(mask.any()):
            continue
        selected_dst = local_dst[mask]
        alpha = weight[mask] / degree_weight_sum[dst[mask]].clamp_min(1e-12)
        source_features = source_feature_getter(src[mask]).to(torch.float32).cpu()
        if source_features.ndim != 2 or int(source_features.shape[1]) != feature_dim:
            raise ValueError("source_feature_getter must return [num_ids, feature_dim]")
        out.index_add_(0, selected_dst, source_features * alpha.unsqueeze(1))

    return StreamingSpMMResult(
        block=out,
        diagnostics={
            "normalization": "destination_row",
            "edge_scans": 2,
            "full_edge_scans": 2,
            "num_edges": int(num_edges),
            "max_edge_chunk_size": int(max_chunk),
            "uses_e_by_d_materialization": False,
            "materialized_full_e_by_d": False,
            "materialized_stacked_edge_index": False,
            "wall_time_s": float(time.perf_counter() - started),
        },
    )
