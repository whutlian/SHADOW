from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest

from t35_fixtures import make_toy_papers100m_root


def test_t35_edge_cache_degree_counts_match_toy_graph(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)

    manifest = build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)

    assert manifest["edge_cache_id"]
    assert manifest["full_edge_scans_for_edge_cache"] == 1
    assert manifest["edge_chunks"] == 3
    dst_degree = np.memmap(cache_root / "graph" / "dst_degree.u32.memmap", mode="r", dtype=np.uint32, shape=(6,))
    src_degree = np.memmap(cache_root / "graph" / "src_degree.u32.memmap", mode="r", dtype=np.uint32, shape=(6,))
    assert dst_degree.tolist() == [0, 3, 0, 1, 2, 0]
    assert src_degree.tolist() == [1, 1, 2, 0, 1, 1]
    assert manifest["edge_cache_bytes"] > 0
    assert manifest["src_degree_checksum"]
    assert manifest["dst_degree_checksum"]
