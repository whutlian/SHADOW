import numpy as np
import pytest
import torch

from shadow_hgc.data.edge_stream import ArrayEdgeStream
from shadow_hgc.data.memmap import create_memmap_feature_store, source_id_block_gather
from shadow_hgc.demand.cache import (
    build_relation_demand_cache,
    estimate_ultra_dry_run,
    validate_train_target_only_cache,
)


def test_large_mode_refuses_all_node_demand_cache_unless_debug():
    with pytest.raises(ValueError, match="all-node"):
        validate_train_target_only_cache(
            num_target_nodes=10,
            train_target_ids=torch.arange(10),
            cache_all_targets=True,
            debug_allow_all_node_cache=False,
        )

    validate_train_target_only_cache(
        num_target_nodes=10,
        train_target_ids=torch.arange(10),
        cache_all_targets=True,
        debug_allow_all_node_cache=True,
    )


def test_streaming_cache_uses_two_scans_and_compact_train_edge_slice(tmp_path):
    src = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    dst = np.array([0, 0, 1, 2, 1], dtype=np.int64)
    features = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [2.0, 2.0]],
        dtype=torch.float32,
    )
    stream_factory = lambda: ArrayEdgeStream(src, dst, chunk_size=2)

    cache = build_relation_demand_cache(
        edge_stream_factory=stream_factory,
        train_target_ids=torch.tensor([0, 1], dtype=torch.long),
        num_target_nodes=3,
        num_source_nodes=4,
        source_feature_getter=lambda ids: features[ids],
        feature_dim=2,
        source_is_target=True,
        cache_edge_slice=True,
    )

    assert cache.stats.full_edge_scans == 2
    assert cache.demand.shape == (2, 2)
    assert cache.edge_slice is not None
    assert cache.edge_slice.num_edges == 2
    assert cache.edge_slice.nbytes > 0


def test_memmap_source_id_block_gather_preserves_request_order(tmp_path):
    path = tmp_path / "features.npy"
    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    store = create_memmap_feature_store(path, data)

    gathered, stats = source_id_block_gather(store, np.array([5, 1, 2, 6]), block_size=2)

    assert np.array_equal(gathered, data[[5, 1, 2, 6]])
    assert stats["num_blocks"] <= 3


def test_ultra_dry_run_estimator_reports_expected_bytes_and_scans():
    estimate = estimate_ultra_dry_run(
        num_train_targets=100,
        num_relations=3,
        feature_dim=128,
        active_source_count=1000,
        train_train_edges=5000,
        dtype_bytes=4,
    )

    assert estimate["demand_cache_bytes"] == 100 * 3 * 128 * 4
    assert estimate["expected_full_edge_scans"] == 6
    assert estimate["edge_slice_cache_bytes"] == 5000 * 12
