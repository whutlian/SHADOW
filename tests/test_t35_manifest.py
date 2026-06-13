from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest

from t35_fixtures import make_toy_papers100m_root


def test_t35_target_local_id_maps_target_idx_and_non_targets(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"

    manifest = build_papers100m_manifest(data_root, cache_root, allow_toy=True)

    assert manifest["num_nodes"] == 6
    assert manifest["num_edges"] == 6
    assert manifest["feature_dim"] == 3
    assert manifest["num_classes"] == 3
    assert manifest["train_size"] == 2
    assert manifest["valid_size"] == 1
    assert manifest["test_size"] == 1
    assert manifest["target_universe_size"] == 4
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(4,))
    target_local_id = np.memmap(cache_root / "raw" / "target_local_id.i32.memmap", mode="r", dtype=np.int32, shape=(6,))
    assert target_idx.tolist() == [0, 1, 2, 4]
    assert target_local_id.tolist() == [0, 1, 2, -1, 3, -1]
    assert manifest["target_idx_checksum"]
    assert manifest["split_checksum"]


def test_t35_manifest_converts_nan_labels_to_minus_one_before_int16_cache(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    labels = np.array([0.0, 1.0, 2.0, np.nan, 1.0, np.nan], dtype=np.float32)
    np.save(data_root / "node_label.npy", labels)
    cache_root = tmp_path / "cache"

    build_papers100m_manifest(data_root, cache_root, allow_toy=True)

    cached = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(6,))
    assert cached.tolist() == [0, 1, 2, -1, 1, -1]
