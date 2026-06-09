from __future__ import annotations

from shadow_hgc.fullgraph.t0s_gates import required_scalability_fields, validate_scalability_resource_row


def test_t0s_scalability_resource_row_requires_bounded_cache_fields():
    row = {
        "num_nodes": 100,
        "num_edges": 1000,
        "num_target_rows": 10,
        "num_train_target_rows": 5,
        "num_active_sources": 20,
        "num_classes": 3,
        "feature_dim": 16,
        "scap_topk": 8,
        "full_edge_scans": 2,
        "peak_cpu_ram_gb": 1.0,
        "peak_gpu_ram_gb": 0.0,
        "disk_cache_gb": 0.1,
        "scap_cache_gb": 0.01,
        "feature_demand_cache_gb": 0.02,
        "wall_time_s": 0.5,
        "edge_scan_throughput_edges_per_s": 1000.0,
        "cache_all_targets": False,
        "uses_dense_e_by_d": False,
    }

    result = validate_scalability_resource_row(row)

    assert result["valid"] is True
    assert set(required_scalability_fields()).issubset(row)


def test_t0s_scalability_resource_row_rejects_all_target_cache():
    row = {field: 1 for field in required_scalability_fields()}
    row["cache_all_targets"] = True
    row["uses_dense_e_by_d"] = False

    result = validate_scalability_resource_row(row)

    assert result["valid"] is False
    assert "cache_all_targets_forbidden" in result["reasons"]
