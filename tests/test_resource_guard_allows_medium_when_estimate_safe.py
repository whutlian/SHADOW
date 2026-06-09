from __future__ import annotations

from shadow_hgc.fullgraph.sfb_v2_train import should_run_medium_row


def test_resource_guard_allows_medium_when_estimate_is_under_limit():
    decision = should_run_medium_row(
        dataset="ogbn-arxiv",
        estimated_cache_bytes=100 * 1024**2,
        memory_limit_bytes=4 * 1024**3,
    )

    assert decision["should_run"] is True
    assert decision["status"] == "run_allowed"
