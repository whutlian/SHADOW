from __future__ import annotations

from shadow_hgc.preprop.block_budget import estimate_block_budget, select_blocks_for_budget


def test_block_budget_reports_all_and_train_target_cache_modes():
    rows = estimate_block_budget(
        dataset="ogbn-papers100M",
        num_target_nodes=1000,
        num_train_target_nodes=100,
        num_edges=5000,
        num_classes=40,
        feature_dim=64,
        selected_blocks=("X0", "X1", "X2", "Y1", "Y2", "structure"),
    )

    modes = {row["cache_mode"]: row for row in rows}
    assert set(modes) == {"all_target_rows", "train_target_only"}
    assert modes["train_target_only"]["total_cache_bytes"] < modes["all_target_rows"]["total_cache_bytes"]
    assert modes["all_target_rows"]["uses_logits_as_input"] is False
    assert modes["all_target_rows"]["uses_e_by_d_materialization"] is False


def test_select_blocks_for_budget_disables_optional_x3_when_too_large():
    selected = select_blocks_for_budget(
        requested_blocks=("X0", "X1", "X2", "X3", "Y1", "Y2", "structure"),
        num_rows=1_000_000,
        feature_dim=128,
        num_classes=40,
        max_cache_gb=0.2,
    )

    assert "X0" in selected
    assert "X3" not in selected
