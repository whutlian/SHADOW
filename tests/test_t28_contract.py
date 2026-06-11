from __future__ import annotations

from shadow_hgc.sft.t28_contract import (
    ARXIV_TEACHER_FIELDS,
    REDDIT_STRUCTURE_FIELDS,
    apply_t28_promotion_guard,
    make_arxiv_teacher_row,
    make_reddit_structure_row,
    summarize_t28_rows,
    validate_t28_promoted_row,
)


def test_t28_required_fields_cover_arxiv_and_reddit_contracts():
    for field in [
        "uses_cns_postprocess",
        "uses_teacher_logits_for_condensation",
        "uses_valid_labels_as_input",
        "teacher_gate_A1_passed",
        "cns_correction_alpha",
    ]:
        assert field in ARXIV_TEACHER_FIELDS
    for field in [
        "prototype_selector",
        "edge_builder",
        "student_model",
        "edge_weight_normalization",
        "uses_processed_data_pt",
        "full_edge_scans",
    ]:
        assert field in REDDIT_STRUCTURE_FIELDS


def test_t28_arxiv_condensation_cannot_promote_before_a1():
    row = make_arxiv_teacher_row(
        method="arxiv_sft_cns",
        seed=42,
        accuracy=0.714,
        macro_f1=0.51,
        predicted_classes=40,
        status="completed_long",
        promotion_status="promoted",
    )
    guarded = apply_t28_promotion_guard(row, dataset_gate_passed=True)
    assert guarded["promotion_status"] == "blocked_teacher_gate"
    assert guarded["promotion_allowed"] is False
    assert "arxiv_teacher_below_A1" in guarded["failure_reason"]


def test_t28_upper_bound_diagnostic_never_promotes_as_scalable_main():
    row = make_arxiv_teacher_row(
        method="arxiv_gnn_teacher_upper_bound",
        seed=42,
        accuracy=0.735,
        macro_f1=0.55,
        predicted_classes=40,
        uses_fullgraph_gnn_teacher=True,
        upper_bound_diagnostic=True,
        status="completed_long",
        promotion_status="promoted",
    )
    guarded = apply_t28_promotion_guard(row, dataset_gate_passed=True)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "upper_bound_diagnostic_not_scalable_main" in guarded["failure_reason"]


def test_t28_reddit_promoted_row_rejects_forbidden_flags_and_bad_ratio():
    row = make_reddit_structure_row(
        method="reddit_sft_knn_graph",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=10,
        condensed_edges=40,
        accuracy=0.917,
        macro_f1=0.89,
        predicted_classes=41,
        status="completed_long",
        promotion_status="promoted",
        uses_processed_data_pt=True,
    )
    result = validate_t28_promoted_row(row)
    assert result["valid"] is False
    assert "uses_processed_data_pt" in result["forbidden_flags"]
    guarded = apply_t28_promotion_guard(row, dataset_gate_passed=True)
    assert guarded["promotion_status"] == "blocked_forbidden"


def test_t28_summary_counts_safe_promoted_rows():
    safe = make_reddit_structure_row(
        method="reddit_sft_knn_graph",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=1,
        condensed_edges=1,
        accuracy=0.917,
        macro_f1=0.89,
        predicted_classes=41,
        status="completed_long",
        promotion_status="promoted",
    )
    blocked = make_reddit_structure_row(
        method="reddit_sft_knn_graph",
        seed=42,
        requested_full_node_ratio=0.001,
        original_num_nodes=1000,
        target_prototypes=1,
        condensed_edges=1,
        accuracy="",
        macro_f1="",
        predicted_classes="",
        status="ready",
        promotion_status="not_promoted",
    )
    summary = summarize_t28_rows([safe, blocked])
    assert summary["rows"] == 2
    assert summary["promoted_rows"] == 1
    assert summary["all_promoted_safe"] is True
