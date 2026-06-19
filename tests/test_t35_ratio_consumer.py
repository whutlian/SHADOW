from __future__ import annotations

import json

import numpy as np

from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table
from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sft_cache import build_or_load_sft_cache
from shadow_hgc.ultra.papers100m_stt_bank import StreamingSTTBankBuilder
from shadow_hgc.ultra.papers100m_teacher import write_teacher_topk_cache_from_probs

from t35_fixtures import make_toy_papers100m_root


def test_t35_ratio_consumer_reuses_cache_ids_and_does_not_scan_edges(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)
    build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)
    build_or_load_sft_cache(cache_root, chunk_size_edges=2, force=True)
    write_teacher_topk_cache_from_probs(
        cache_root,
        np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.15], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]], dtype=np.float32),
        mode="topk2_tail",
    )
    ctx = Papers100MCacheContext(cache_root)
    StreamingSTTBankBuilder(ctx, policy="stt_ratio_v2", seed=42, max_ratio=0.75, chunk_size=2).build_bank()
    ctx = Papers100MCacheContext(cache_root)
    before_ids = ctx.cache_ids()

    row_a = materialize_condensed_table(ctx, 0.25)
    row_b = materialize_condensed_table(ctx, 0.75)

    for row in [row_a, row_b]:
        assert row["cache_reused"] is True
        assert row["edge_slice_cache_reused"] is True
        assert row["sft_cache_reused"] is True
        assert row["teacher_cache_reused"] is True
        assert row["selection_bank_reused"] is True
        assert row["incremental_edge_scans_after_cache_build"] == 0
        assert row["edge_slice_cache_id"] == before_ids["edge_slice_cache_id"]
        assert row["sft_cache_id"] == before_ids["sft_cache_id"]
        assert row["teacher_cache_id"] == before_ids["teacher_cache_id"]
        assert row["selection_bank_id"] == before_ids["selection_bank_id"]
    assert row_a["condensed_nodes"] == 2
    assert row_b["condensed_nodes"] == 4
    assert row_a["full_node_ratio_denominator"] == 6
    labels = np.memmap(
        cache_root / "condensed" / "policy=stt_ratio_v2_seed42" / "ratio=0.75" / "hard_anchor_labels.int16.memmap",
        mode="r",
        dtype=np.int16,
        shape=(4,),
    )
    assert -1 in labels.tolist()
    assert (cache_root / "condensed" / "policy=stt_ratio_v2_seed42" / "ratio=0.25" / "condensed_manifest.json").exists()
    assert (cache_root / "condensed" / "policy=stt_ratio_v2_seed42" / "ratio=0.75" / "condensed_manifest.json").exists()


def test_t35_condensed_materialization_is_seed_isolated(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)
    build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)
    build_or_load_sft_cache(cache_root, chunk_size_edges=2, force=True)
    write_teacher_topk_cache_from_probs(
        cache_root,
        np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.15], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]], dtype=np.float32),
        mode="topk2_tail",
    )
    ctx42 = Papers100MCacheContext(cache_root, selection_policy="stt_ratio_v2", seed=42)
    StreamingSTTBankBuilder(ctx42, policy="stt_ratio_v2", seed=42, max_ratio=0.75, chunk_size=2).build_bank()
    ctx43 = Papers100MCacheContext(cache_root, selection_policy="stt_ratio_v2", seed=43)
    StreamingSTTBankBuilder(ctx43, policy="stt_ratio_v2", seed=43, max_ratio=0.75, chunk_size=2).build_bank()
    ctx42 = Papers100MCacheContext(cache_root, selection_policy="stt_ratio_v2", seed=42)
    ctx43 = Papers100MCacheContext(cache_root, selection_policy="stt_ratio_v2", seed=43)

    row42 = materialize_condensed_table(ctx42, 0.75, policy="stt_ratio_v2", seed=42)
    row43 = materialize_condensed_table(ctx43, 0.75, policy="stt_ratio_v2", seed=43)

    path42 = cache_root / "condensed" / "policy=stt_ratio_v2_seed42" / "ratio=0.75" / "condensed_manifest.json"
    path43 = cache_root / "condensed" / "policy=stt_ratio_v2_seed43" / "ratio=0.75" / "condensed_manifest.json"
    assert path42.exists()
    assert path43.exists()
    manifest42 = json.loads(path42.read_text(encoding="utf-8"))
    manifest43 = json.loads(path43.read_text(encoding="utf-8"))
    assert manifest42["condensed_cache_id"] != manifest43["condensed_cache_id"]
    assert manifest42["condensed_dir"] != manifest43["condensed_dir"]
    assert row42["condensed_nodes"] == row43["condensed_nodes"] == 4
