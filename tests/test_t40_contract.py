from __future__ import annotations

from shadow_hgc.sft.t40_contract import (
    FIXED_CANDIDATE_POLICIES,
    PUBLIC_METHOD_ID,
    PUBLIC_METHOD_NAME,
    make_t40_row,
    validate_t40_main_row,
    validate_t40_main_table,
)


def test_t40_main_method_id_and_public_name_are_frozen() -> None:
    row = make_t40_row(
        dataset="Reddit",
        requested_full_node_ratio=0.001,
        condensed_nodes=233,
        num_classes=41,
        accuracy=0.932,
        macro_f1=0.90,
        valid_acc=0.931,
        selected_policy="teacher_transport",
        promotion_status="promoted",
    )

    result = validate_t40_main_row(row)

    assert row["method"] == PUBLIC_METHOD_ID == "shadow_stt_unified_auto_v2"
    assert row["public_method_name"] == PUBLIC_METHOD_NAME == "Shadow-HGC-STT-U"
    assert result["valid"] is True


def test_t40_main_guard_rejects_legacy_public_method_ids() -> None:
    row = make_t40_row(
        dataset="ogbn-products",
        requested_full_node_ratio=0.005,
        condensed_nodes=12_245,
        num_classes=47,
        method="products_uca_hybrid_mixup",
        accuracy=0.77,
        macro_f1=0.42,
        valid_acc=0.89,
        selected_policy="domain_coverage",
        promotion_status="promoted",
    )

    result = validate_t40_main_row(row)

    assert result["valid"] is False
    assert "non_unified_method_id_in_main_table" in result["forbidden_flags"]
    assert "legacy_specialized_method_id_in_main_table" in result["forbidden_flags"]


def test_t40_promoted_rows_reject_forbidden_paths() -> None:
    row = make_t40_row(
        dataset="ogbn-papers100M",
        requested_full_node_ratio=0.001,
        condensed_nodes=111_060,
        num_classes=172,
        accuracy=0.60,
        macro_f1=0.40,
        valid_acc=0.64,
        selected_policy="high_fidelity",
        promotion_status="promoted",
        uses_teacher_probs_as_input_features=True,
        uses_dense_all_node_teacher_cache=True,
        uses_full_edge_index_on_gpu=True,
    )

    result = validate_t40_main_row(row)

    assert result["valid"] is False
    assert "uses_teacher_probs_as_input_features" in result["forbidden_flags"]
    assert "uses_dense_all_node_teacher_cache" in result["forbidden_flags"]
    assert "uses_full_edge_index_on_gpu" in result["forbidden_flags"]


def test_t40_candidate_policy_set_is_fixed_metadata_not_method_name() -> None:
    assert FIXED_CANDIDATE_POLICIES == (
        "auto_base",
        "coverage_heavy",
        "domain_coverage",
        "teacher_transport",
        "high_fidelity",
    )

    row = make_t40_row(
        dataset="Reddit",
        requested_full_node_ratio=0.005,
        condensed_nodes=1165,
        num_classes=41,
        selected_policy="high_fidelity",
        policy_candidate_count=len(FIXED_CANDIDATE_POLICIES),
        accuracy=0.939,
        macro_f1=0.90,
        valid_acc=0.938,
        promotion_status="promoted",
    )

    assert row["method"] == PUBLIC_METHOD_ID
    assert row["selected_policy"] == "high_fidelity"
    assert validate_t40_main_row(row)["valid"] is True


def test_t40_papers100m_table_requires_one_cache_ids_and_zero_incremental_scans() -> None:
    rows = [
        make_t40_row(
            dataset="ogbn-papers100M",
            requested_full_node_ratio=ratio,
            condensed_nodes=nodes,
            num_classes=172,
            accuracy=0.61,
            macro_f1=0.40,
            valid_acc=0.65,
            selected_policy="high_fidelity",
            edge_cache_id="edge-a",
            sft_cache_id="sft-a",
            teacher_cache_id="teacher-a",
            reservoir_cache_id="reservoir-a",
            cache_reused=True,
            incremental_edge_scans_after_cache_build=0,
            promotion_status="promoted",
        )
        for ratio, nodes in [(0.0005, 55_530), (0.001, 111_060)]
    ]

    assert validate_t40_main_table(rows)["valid"] is True
    rows[1]["reservoir_cache_id"] = "reservoir-b"

    result = validate_t40_main_table(rows)

    assert result["valid"] is False
    assert "papers100m_reservoir_cache_id_mismatch" in result["forbidden_flags"]


def test_t40_table_audit_ignores_non_papers_rows() -> None:
    row = make_t40_row(
        dataset="Reddit",
        requested_full_node_ratio=0.001,
        condensed_nodes=233,
        num_classes=41,
        accuracy=0.92,
        macro_f1=0.88,
        valid_acc=0.92,
        promotion_status="promoted",
        cache_reused=False,
        edge_cache_id="",
        sft_cache_id="",
        teacher_cache_id="",
        reservoir_cache_id="",
    )

    result = validate_t40_main_table([row])

    assert result["valid"] is True


def test_t40_table_audit_ignores_blocked_papers_rows() -> None:
    reddit = make_t40_row(
        dataset="Reddit",
        requested_full_node_ratio=0.001,
        condensed_nodes=233,
        num_classes=41,
        accuracy=0.92,
        macro_f1=0.88,
        valid_acc=0.92,
        promotion_status="promoted",
        cache_reused=True,
    )
    papers_blocked = make_t40_row(
        dataset="ogbn-papers100M",
        requested_full_node_ratio=0.0001,
        condensed_nodes=11106,
        num_classes=172,
        promotion_status="blocked",
        failure_reason="all_candidate_policies_blocked",
    )

    result = validate_t40_main_table([reddit, papers_blocked])

    assert result["valid"] is True
