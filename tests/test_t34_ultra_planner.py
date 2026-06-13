from __future__ import annotations

from shadow_hgc.sft.ultra_stt_planner import plan_ultra_stt


def test_t34_ultra_stt_promotes_topk_tail_and_rejects_dense() -> None:
    topk = plan_ultra_stt(
        dataset="ogbn-papers100M",
        num_nodes=111_059_956,
        num_edges=1_615_685_872,
        num_classes=172,
        requested_ratio=0.0001,
        teacher_cache_mode="topk8_tail",
        signature_dim=128,
    )
    assert topk["promotion_status"] == "promoted"
    assert topk["uses_dense_nxc_teacher_cache"] is False
    assert topk["uses_all_pair"] is False
    assert topk["planned_condensed_nodes"] == 11106

    dense = plan_ultra_stt(
        dataset="MAG240M",
        num_nodes=121_751_666,
        num_edges=1_728_364_232,
        num_classes=153,
        requested_ratio=0.0001,
        teacher_cache_mode="dense_fp16",
        signature_dim=64,
    )
    assert dense["promotion_status"] == "blocked_forbidden"
    assert "ultra_dense_nxc_teacher_cache" in dense["failure_reason"]
