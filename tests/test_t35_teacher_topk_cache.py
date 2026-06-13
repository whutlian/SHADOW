from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_teacher import load_teacher_topk_cache, write_teacher_topk_cache_from_probs, write_teacher_topk_cache_from_prototypes


def test_t35_topk_teacher_cache_reconstructs_probability_distribution(tmp_path):
    cache_root = tmp_path / "cache"
    probs = np.array(
        [
            [0.65, 0.20, 0.10, 0.05],
            [0.05, 0.10, 0.75, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )

    manifest = write_teacher_topk_cache_from_probs(cache_root, probs, mode="topk2_tail")
    cache = load_teacher_topk_cache(cache_root)
    dense = cache.reconstruct_rows(np.array([0, 1, 2], dtype=np.int64), num_classes=4)

    assert manifest["teacher_cache_mode"] == "topk2_tail"
    assert manifest["teacher_cache_scope"] == "target_universe"
    assert dense.shape == probs.shape
    assert np.allclose(dense.sum(axis=1), np.ones(3), atol=1e-4)
    assert np.all(cache.tail_mass >= 0)
    assert manifest["uses_dense_all_node_teacher_cache"] is False


def test_t35_streaming_teacher_writer_does_not_mark_dense_teacher_ram(tmp_path):
    cache_root = tmp_path / "cache"
    features = np.memmap(tmp_path / "features.fp16.memmap", mode="w+", dtype=np.float16, shape=(4, 3))
    features[:] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float16,
    )
    features.flush()
    prototypes = np.eye(3, dtype=np.float32)

    manifest = write_teacher_topk_cache_from_prototypes(cache_root, features, prototypes, mode="topk2_tail", chunk_size=2)

    assert manifest["teacher_topk_build_mode"] == "streaming_logits"
    assert manifest["uses_dense_teacher_cache_in_ram"] is False
    assert manifest["uses_dense_all_node_teacher_cache"] is False
