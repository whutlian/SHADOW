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
    assert cache.stats.edge_slice_cache_edges == 2
    assert cache.stats.edge_slice_cache_bytes == cache.edge_slice.nbytes
    assert cache.stats.edge_slice_dtype == "int32,int32,float32"
    assert cache.stats.cache_build_time >= 0.0
    assert cache.stats.cache_aggregation_time >= 0.0
    assert cache.stats.disk_spill_used is False


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


def test_ultra_dry_run_estimator_reports_relation_specific_memory_and_index_mode():
    estimate = estimate_ultra_dry_run(
        num_train_targets=100,
        feature_dim=16,
        dtype_bytes=4,
        dense_map_budget_bytes=1_000,
        relations=[
            {
                "name": "paper--cite_ref-->paper",
                "num_edges": 1_000,
                "num_train_target_incident_edges": 300,
                "num_train_train_edges": 40,
                "num_active_sources": 25,
                "num_source_nodes": 1_000_000,
                "num_target_nodes": 1_000_000,
                "source_is_target": True,
            },
            {
                "name": "author--writes-->paper",
                "num_edges": 2_000,
                "num_train_target_incident_edges": 500,
                "num_train_train_edges": 0,
                "num_active_sources": 80,
                "num_source_nodes": 2_000_000,
                "num_target_nodes": 1_000_000,
                "source_is_target": False,
            },
        ],
    )

    assert estimate["total_expected_full_edge_scans"] == 4
    assert estimate["expected_full_edge_scans"] == 4
    cite = estimate["relations"]["paper--cite_ref-->paper"]
    writes = estimate["relations"]["author--writes-->paper"]
    assert cite["demand_cache_bytes"] == 100 * 16 * 4
    assert cite["edge_slice_cache_edges"] == 40
    assert cite["edge_slice_cache_bytes"] == 40 * 12
    assert cite["id_index_mode"] == "sorted_search"
    assert writes["edge_slice_cache_bytes"] == 0
    assert writes["active_source_feature_bytes"] == 80 * 16 * 4


def test_streaming_cache_can_use_sorted_id_index_for_large_node_space():
    src = np.array([0, 1, 2, 3], dtype=np.int64)
    dst = np.array([10, 20, 11, 20], dtype=np.int64)
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    cache = build_relation_demand_cache(
        edge_stream_factory=lambda: ArrayEdgeStream(src, dst, chunk_size=2),
        train_target_ids=torch.tensor([10, 20], dtype=torch.long),
        num_target_nodes=1_000,
        num_source_nodes=6,
        source_feature_getter=lambda ids: features[ids],
        feature_dim=2,
        source_is_target=False,
        dense_map_budget_bytes=16,
    )

    assert cache.stats.dst_id_index_mode == "sorted_search"
    assert cache.stats.src_train_id_index_mode is None
    assert cache.stats.full_edge_scans == 2
