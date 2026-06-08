from __future__ import annotations

import numpy as np

from shadow_hgc.features.streaming_diffusion import compute_streaming_diffusion_blocks


class GuardedProvider:
    def __init__(self, *, num_nodes: int, dim: int, max_batch: int) -> None:
        self.num_nodes = num_nodes
        self.dim = dim
        self.max_batch = max_batch
        self.max_seen = 0
        self.calls = 0

    def get(self, indices):
        self.calls += 1
        batch = len(indices)
        self.max_seen = max(self.max_seen, batch)
        if batch > self.max_batch:
            raise AssertionError(f"unbounded feature gather: {batch} > {self.max_batch}")
        rows = np.asarray(indices, dtype=np.float32)[:, None]
        cols = np.arange(self.dim, dtype=np.float32)[None, :]
        return rows * 0.01 + cols


def test_streaming_diffusion_uses_bounded_edge_chunks_for_large_graph(tmp_path):
    num_nodes = 2000
    num_edges = 5000
    dim = 16
    chunk_size = 127
    src = np.arange(num_edges, dtype=np.int64) % num_nodes
    dst = (np.arange(num_edges, dtype=np.int64) * 17) % num_nodes
    edge_index = np.stack([src, dst], axis=0)
    provider = GuardedProvider(num_nodes=num_nodes, dim=dim, max_batch=chunk_size)

    result = compute_streaming_diffusion_blocks(
        x_provider=provider,
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=(1,),
        include_highpass=False,
        out_dir=tmp_path,
        dtype="float16",
        edge_chunk_size=chunk_size,
        overwrite=True,
    )

    assert provider.max_seen <= chunk_size
    assert provider.calls > 1
    assert result.block_shapes["X1"] == (num_nodes, dim)
    assert result.stats["edge_chunk_size"] == chunk_size
    assert result.stats["full_edge_scans"] == 1
