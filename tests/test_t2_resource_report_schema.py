from shadow_hgc.preprop.manifest import validate_t2_resource_report


def test_t2_resource_report_requires_scalability_flags():
    row = {
        "full_edge_scans": 2,
        "edge_chunk_size": 1024,
        "dst_chunk_size": 128,
        "block_dim": 64,
        "num_blocks": 3,
        "cache_bytes": 4096,
        "peak_cpu_ram_gb": 0.25,
        "peak_gpu_ram_gb": 0.0,
        "wall_time_s": 1.0,
        "uses_memmap": True,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_logits_as_input": False,
        "uses_bounded_edges": False,
    }

    result = validate_t2_resource_report(row)

    assert result["valid"] is True
    assert result["missing_fields"] == []
