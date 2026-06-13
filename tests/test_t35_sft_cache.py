from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest
from shadow_hgc.ultra.papers100m_sft_cache import build_or_load_sft_cache

from t35_fixtures import make_toy_papers100m_root, toy_expected_cite_ref_x1, toy_expected_cited_by_x1


def test_t35_sft_x1_target_blocks_match_dense_reference(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)
    build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)

    manifest = build_or_load_sft_cache(cache_root, chunk_size_edges=2, force=True, x2_mode="disabled")

    assert manifest["sft_cache_id"]
    assert manifest["x2_mode"] == "disabled"
    assert set(manifest["blocks"]) >= {
        "X0_target",
        "X1_cite_ref_target",
        "X1_cited_by_target",
        "degree_target",
        "label_support_target",
        "label_entropy_target",
    }
    x1_ref = np.memmap(cache_root / "sft" / "X1_cite_ref_target.fp16.memmap", mode="r", dtype=np.float16, shape=(4, 3))
    x1_rev = np.memmap(cache_root / "sft" / "X1_cited_by_target.fp16.memmap", mode="r", dtype=np.float16, shape=(4, 3))
    assert np.allclose(np.asarray(x1_ref, dtype=np.float32), toy_expected_cite_ref_x1(), atol=5e-4)
    assert np.allclose(np.asarray(x1_rev, dtype=np.float32), toy_expected_cited_by_x1(), atol=5e-4)
    assert manifest["sft_cache_bytes"] > 0
