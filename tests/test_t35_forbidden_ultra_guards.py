from __future__ import annotations

from shadow_hgc.ultra.papers100m_contract import T35_REQUIRED_FIELDS, make_t35_row, validate_t35_row


def test_t35_promoted_row_fails_for_dense_all_node_teacher_cache():
    row = make_t35_row(
        promotion_status="promoted",
        uses_dense_teacher_cache_in_ram=True,
        teacher_cache_scope="all_nodes",
    )

    result = validate_t35_row(row)

    assert result["valid"] is False
    assert "uses_dense_teacher_cache_in_ram" in result["forbidden_flags"]
    assert "teacher_cache_scope_not_target_universe" in result["forbidden_flags"]


def test_t35_promoted_row_fails_for_full_edge_gpu_and_all_pair_paths():
    row = make_t35_row(
        promotion_status="promoted",
        uses_full_edge_index_on_gpu=True,
        uses_e_by_d_materialization=True,
        uses_dense_p2=True,
        uses_all_pair_distance=True,
    )

    result = validate_t35_row(row)

    assert result["valid"] is False
    assert "uses_full_edge_index_on_gpu" in result["forbidden_flags"]
    assert "uses_e_by_d_materialization" in result["forbidden_flags"]
    assert "uses_dense_p2" in result["forbidden_flags"]
    assert "uses_all_pair_distance" in result["forbidden_flags"]


def test_t35_row_schema_contains_required_cache_reuse_fields():
    row = make_t35_row()
    missing = [field for field in T35_REQUIRED_FIELDS if field not in row]
    assert missing == []
    assert row["uses_valid_labels_as_input"] is False
    assert row["uses_test_labels_as_input"] is False
    assert row["uses_teacher_probs_as_input"] is False
    assert row["uses_teacher_probs_as_soft_targets"] is True
