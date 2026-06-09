from __future__ import annotations

from shadow_hgc.fullgraph.sfb_v2_train import format_allocation_failure


def test_resource_guard_reports_allocation_shape_and_module():
    report = format_allocation_failure(
        tensor_shape=(2449029, 512),
        requested_bytes=2449029 * 512 * 4,
        chunk_size=65536,
        current_cache_bytes=128,
        peak_ram_gb=12.5,
        module_name="typed_feature_demand",
    )

    assert report["tensor_shape"] == [2449029, 512]
    assert report["requested_bytes"] == 2449029 * 512 * 4
    assert report["chunk_size"] == 65536
    assert report["module_name"] == "typed_feature_demand"
