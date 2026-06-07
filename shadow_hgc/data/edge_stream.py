from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass
class EdgeChunk:
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor


class ArrayEdgeStream:
    """Re-iterable chunked edge stream over local arrays."""

    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        weight: np.ndarray | None = None,
        *,
        chunk_size: int = 1_000_000,
    ) -> None:
        if src.shape != dst.shape:
            raise ValueError("src and dst arrays must have the same shape")
        self.src = src
        self.dst = dst
        self.weight = weight
        self.chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[EdgeChunk]:
        for start in range(0, len(self.src), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.src))
            src = torch.as_tensor(self.src[start:stop], dtype=torch.long)
            dst = torch.as_tensor(self.dst[start:stop], dtype=torch.long)
            if self.weight is None:
                weight = torch.ones(stop - start, dtype=torch.float32)
            else:
                weight = torch.as_tensor(self.weight[start:stop], dtype=torch.float32)
            yield EdgeChunk(src=src, dst=dst, weight=weight)


class SyntheticEdgeStream:
    """Deterministic synthetic edge stream for stress and dry-run tests."""

    def __init__(
        self,
        *,
        num_edges: int,
        num_src_nodes: int,
        num_dst_nodes: int,
        chunk_size: int = 1_000_000,
        seed: int = 0,
    ) -> None:
        self.num_edges = int(num_edges)
        self.num_src_nodes = int(num_src_nodes)
        self.num_dst_nodes = int(num_dst_nodes)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[EdgeChunk]:
        rng = np.random.default_rng(self.seed)
        remaining = self.num_edges
        while remaining > 0:
            n = min(self.chunk_size, remaining)
            src = torch.as_tensor(rng.integers(0, self.num_src_nodes, size=n), dtype=torch.long)
            dst = torch.as_tensor(rng.integers(0, self.num_dst_nodes, size=n), dtype=torch.long)
            yield EdgeChunk(src=src, dst=dst, weight=torch.ones(n, dtype=torch.float32))
            remaining -= n


def run_synthetic_streaming_stress(
    *,
    output_path: str | Path,
    num_edges: int,
    num_src_nodes: int,
    num_dst_nodes: int,
    num_train_targets: int,
    feature_dim: int,
    chunk_size: int = 1_000_000,
    seed: int = 0,
) -> dict:
    """Run a bounded synthetic two-pass streaming cache stress test."""

    from shadow_hgc.demand.cache import build_relation_demand_cache
    from shadow_hgc.eval.logging import write_json_summary

    stream_factory = lambda: SyntheticEdgeStream(
        num_edges=num_edges,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        chunk_size=chunk_size,
        seed=seed,
    )

    def source_feature_getter(ids: torch.Tensor) -> torch.Tensor:
        cols = torch.arange(feature_dim, dtype=torch.float32).unsqueeze(0)
        return torch.sin(ids.to(torch.float32).unsqueeze(1) * 0.017 + cols * 0.13)

    train_target_ids = torch.arange(min(num_train_targets, num_dst_nodes), dtype=torch.long)
    cache = build_relation_demand_cache(
        edge_stream_factory=stream_factory,
        train_target_ids=train_target_ids,
        num_target_nodes=num_dst_nodes,
        num_source_nodes=num_src_nodes,
        source_feature_getter=source_feature_getter,
        feature_dim=feature_dim,
        source_is_target=True,
        cache_edge_slice=True,
    )
    summary = {
        "dataset": "synthetic_streaming_stress",
        "num_edges": int(num_edges),
        "num_src_nodes": int(num_src_nodes),
        "num_dst_nodes": int(num_dst_nodes),
        "num_train_targets": int(train_target_ids.numel()),
        "feature_dim": int(feature_dim),
        "chunk_size": int(chunk_size),
        "full_edge_scans": cache.stats.full_edge_scans,
        "active_source_count": cache.stats.active_source_count,
        "edge_slice_cache_bytes": cache.stats.edge_slice_cache_bytes,
        "demand_shape": list(cache.demand.shape),
        "cache_all_targets": False,
    }
    write_json_summary(output_path, summary)
    return summary
