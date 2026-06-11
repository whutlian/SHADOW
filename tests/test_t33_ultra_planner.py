from __future__ import annotations

from shadow_hgc.sft.ultra_ttcpp_planner import plan_ultra_ttcpp


def test_t33_ultra_planner_uses_topk_and_estimates_resources() -> None:
    row = plan_ultra_ttcpp(
        dataset="ogbn-papers100M",
        num_nodes=111_059_956,
        num_edges=1_615_685_872,
        num_classes=172,
        requested_ratio=0.0001,
        teacher_cache_mode="topk8_fp16",
        signature_dim=128,
    )
    assert row["planned_condensed_nodes"] == 11106
    assert row["teacher_topk_cache_bytes"] < row["teacher_dense_cache_bytes_diagnostic"]
    assert row["uses_dense_teacher_cache_in_ram"] is False
    assert row["uses_all_pair_distance"] is False
    assert row["promotion_status"] == "promoted"


def test_t33_ultra_planner_blocks_dense_cache_promotion() -> None:
    row = plan_ultra_ttcpp(
        dataset="MAG240M",
        num_nodes=121_751_666,
        num_edges=1_728_364_232,
        num_classes=153,
        requested_ratio=0.0001,
        teacher_cache_mode="dense_fp16",
        signature_dim=64,
    )
    assert row["promotion_status"] == "blocked_forbidden"
    assert row["failure_reason"] == "ultra_dense_teacher_cache"
