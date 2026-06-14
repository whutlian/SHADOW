from __future__ import annotations

from shadow_hgc.sft.unified_stt import (
    PUBLIC_METHOD_ID,
    PUBLIC_METHOD_NAME,
    make_t38_row,
    validate_t38_main_row,
    validate_t38_main_table,
)


def test_t38_main_method_name_guard_rejects_specialized_ids() -> None:
    row = make_t38_row(
        dataset="Reddit",
        requested_full_node_ratio=0.001,
        condensed_nodes=233,
        num_classes=41,
        accuracy=0.93,
        macro_f1=0.90,
        method="reddit_stt_gamlp_ratio_v2",
        public_method=PUBLIC_METHOD_NAME,
        promotion_status="promoted",
    )

    result = validate_t38_main_row(row)

    assert result["valid"] is False
    assert "main_method_id_mismatch" in result["forbidden_flags"]
    assert "old_specialized_method_id_in_main" in result["forbidden_flags"]


def test_t38_valid_main_row_has_public_method_only() -> None:
    row = make_t38_row(
        dataset="ogbn-products",
        requested_full_node_ratio=0.0004,
        condensed_nodes=980,
        num_classes=47,
        accuracy=0.70,
        macro_f1=0.33,
        promotion_status="promoted",
    )

    result = validate_t38_main_row(row)

    assert row["method"] == PUBLIC_METHOD_ID
    assert row["public_method"] == PUBLIC_METHOD_NAME
    assert result["valid"] is True


def test_t38_promoted_rows_reject_forbidden_flags() -> None:
    row = make_t38_row(
        dataset="ogbn-papers100M",
        requested_full_node_ratio=0.001,
        condensed_nodes=111_060,
        num_classes=172,
        accuracy=0.60,
        macro_f1=0.40,
        promotion_status="promoted",
        uses_dense_all_node_teacher_cache=True,
        uses_teacher_probs_as_input_features=True,
        uses_valid_labels_as_input=True,
    )

    result = validate_t38_main_row(row)

    assert result["valid"] is False
    assert "uses_dense_all_node_teacher_cache" in result["forbidden_flags"]
    assert "uses_teacher_probs_as_input_features" in result["forbidden_flags"]
    assert "uses_valid_labels_as_input" in result["forbidden_flags"]


def test_t38_table_guard_checks_papers100m_one_cache_reuse() -> None:
    rows = [
        make_t38_row(
            dataset="ogbn-papers100M",
            requested_full_node_ratio=ratio,
            condensed_nodes=nodes,
            num_classes=172,
            accuracy=0.60,
            macro_f1=0.40,
            edge_cache_id="edge-a",
            sft_cache_id="sft-a",
            teacher_cache_id="teacher-a",
            unified_reservoir_id="reservoir-a",
            cache_reused=True,
            incremental_edge_scans_after_cache_build=0,
            promotion_status="promoted",
        )
        for ratio, nodes in [(0.0005, 55_530), (0.001, 111_060)]
    ]

    assert validate_t38_main_table(rows)["valid"] is True
    rows[1]["edge_cache_id"] = "edge-b"

    result = validate_t38_main_table(rows)

    assert result["valid"] is False
    assert "papers100m_edge_cache_id_mismatch" in result["forbidden_flags"]
