from __future__ import annotations

from shadow_hgc.sft.t30_contract import (
    T30_REQUIRED_FIELDS,
    apply_t30_promotion_guard,
    make_t30_row,
    ratio_budget,
    validate_t30_row,
)


def test_t30_schema_contains_qoc_and_forbidden_fields() -> None:
    required = set(T30_REQUIRED_FIELDS)
    for field in [
        "num_codewords",
        "num_labeled_codewords",
        "num_unlabeled_codewords",
        "operator_edges_before_topk",
        "operator_edges_after_topk",
        "quotient_build_mode",
        "transfer_eval_type",
        "uses_processed_data_pt",
    ]:
        assert field in required


def test_t30_ratio_budget_uses_full_original_node_count() -> None:
    assert ratio_budget("Reddit", 0.001) == 233
    assert ratio_budget("Reddit", 0.005) == 1165
    first = make_t30_row(dataset="Reddit", method="Shadow-QOC-hard", seed=42, requested_full_node_ratio=0.001, num_codewords=233)
    second = make_t30_row(dataset="Reddit", method="Shadow-QOC-hard", seed=42, requested_full_node_ratio=0.005, num_codewords=1165)
    assert abs(float(first["actual_full_node_ratio"]) - 0.001) <= 1.0 / float(first["original_num_nodes"])
    assert abs(float(second["actual_full_node_ratio"]) - 0.005) <= 1.0 / float(second["original_num_nodes"])


def test_t30_rejects_reused_96_node_smoke_budget_for_multiple_ratios() -> None:
    row = make_t30_row(
        dataset="Reddit",
        method="Shadow-QOC-hard",
        seed=42,
        requested_full_node_ratio=0.005,
        num_codewords=96,
        status="completed_transfer_eval",
        promotion_status="promoted",
        promotion_track="safe_main",
        accuracy=0.93,
        macro_f1=0.90,
        valid_acc=0.92,
        predicted_classes=41,
        transfer_eval_type="real_transfer_eval",
        extra={"operator_row_sum_error": 0.0},
    )
    result = validate_t30_row(row)
    assert not result["valid"]
    assert "ratio_mismatch" in result["forbidden_flags"]


def test_t30_safe_promotion_guard_rejects_forbidden_flags() -> None:
    row = make_t30_row(
        dataset="Reddit",
        method="Shadow-QOC-hard",
        seed=42,
        requested_full_node_ratio=0.001,
        num_codewords=233,
        status="completed_transfer_eval",
        promotion_status="promoted",
        promotion_track="safe_main",
        accuracy=0.924,
        macro_f1=0.891,
        valid_acc=0.923,
        predicted_classes=41,
        transfer_eval_type="real_transfer_eval",
        extra={"operator_row_sum_error": 0.0},
        uses_teacher_logits=True,
    )
    guarded = apply_t30_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_logits" in guarded["failure_reason"]


def test_t30_sota_still_rejects_valid_and_test_label_inputs() -> None:
    row = make_t30_row(
        dataset="Reddit",
        method="Shadow-QOC-soft",
        seed=42,
        requested_full_node_ratio=0.005,
        num_codewords=1165,
        status="completed_transfer_eval",
        promotion_status="promoted",
        promotion_track="sota_chase",
        accuracy=0.933,
        macro_f1=0.902,
        valid_acc=0.932,
        predicted_classes=41,
        transfer_eval_type="real_transfer_eval",
        extra={"operator_row_sum_error": 0.0},
        uses_teacher_logits=True,
        uses_valid_labels_as_input=True,
    )
    guarded = apply_t30_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_valid_labels_as_input" in guarded["failure_reason"]


def test_t30_qoc_cannot_promote_without_real_transfer_accuracy() -> None:
    row = make_t30_row(
        dataset="Reddit",
        method="Shadow-QOC-hard",
        seed=42,
        requested_full_node_ratio=0.001,
        num_codewords=233,
        status="completed_operator_smoke",
        promotion_status="promoted",
        promotion_track="safe_main",
        failure_reason="no_transfer_eval_accuracy",
        transfer_eval_type="operator_smoke",
        extra={"operator_row_sum_error": 0.0},
    )
    guarded = apply_t30_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "missing_accuracy" in guarded["failure_reason"]
    assert "qoc_requires_real_transfer_eval" in guarded["failure_reason"]


def test_t30_qoc_rejects_bad_operator_row_sum_error() -> None:
    row = make_t30_row(
        dataset="Reddit",
        method="Shadow-QOC-hard",
        seed=42,
        requested_full_node_ratio=0.001,
        num_codewords=233,
        status="completed_transfer_eval",
        promotion_status="promoted",
        promotion_track="safe_main",
        accuracy=0.924,
        macro_f1=0.891,
        valid_acc=0.923,
        predicted_classes=41,
        transfer_eval_type="real_transfer_eval",
        extra={"operator_row_sum_error": 1e-3},
    )
    result = validate_t30_row(row)
    assert not result["valid"]
    assert "operator_row_sum_error" in result["forbidden_flags"]
