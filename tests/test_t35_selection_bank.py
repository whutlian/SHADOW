from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sft_cache import build_or_load_sft_cache
from shadow_hgc.ultra.papers100m_stt_bank import StreamingSTTBankBuilder, load_selection_bank
from shadow_hgc.ultra.papers100m_teacher import write_teacher_topk_cache_from_probs

from t35_fixtures import make_toy_papers100m_root


def test_t35_selection_bank_serves_nested_multi_ratio_prefixes_without_rebuild(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)
    build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)
    build_or_load_sft_cache(cache_root, chunk_size_edges=2, force=True)
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.15], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]], dtype=np.float32)
    write_teacher_topk_cache_from_probs(cache_root, probs, mode="topk2_tail")

    ctx = Papers100MCacheContext(cache_root)
    bank_manifest = StreamingSTTBankBuilder(ctx, policy="stt_ratio_v2", seed=7, max_ratio=0.75, chunk_size=2).build_bank()
    ctx_seed7 = Papers100MCacheContext(cache_root, selection_policy="stt_ratio_v2", seed=7)
    bank = load_selection_bank(cache_root)
    small = bank.select_prefix(0.25, full_node_denominator=4)
    large = bank.select_prefix(0.75, full_node_denominator=4)

    assert bank_manifest["selection_bank_id"]
    assert ctx_seed7.cache_ids()["selection_bank_id"] == bank_manifest["selection_bank_id"]
    assert bank_manifest["nested_selection"] is True
    assert bank_manifest["bank_build_count"] == 1
    assert small.size == 1
    assert large.size == 3
    assert set(small.tolist()).issubset(set(large.tolist()))
    assert bank_manifest["bucket_core_count"] >= 1
