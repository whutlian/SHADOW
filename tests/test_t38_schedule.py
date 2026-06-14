from __future__ import annotations

from shadow_hgc.sft.unified_schedule import (
    compute_teacher_cache_policy,
    compute_unified_schedule,
    full_node_ratio,
)


def test_t38_auto_schedule_is_deterministic() -> None:
    first = compute_unified_schedule(
        condensed_nodes=16_384,
        num_classes=64,
        teacher_valid_acc=0.80,
        majority_valid_acc=0.20,
        num_nodes=100_000,
        num_teacher_nodes=100_000,
    )
    second = compute_unified_schedule(
        condensed_nodes=16_384,
        num_classes=64,
        teacher_valid_acc=0.80,
        majority_valid_acc=0.20,
        num_nodes=100_000,
        num_teacher_nodes=100_000,
    )

    assert first == second
    assert first.budget_per_class == 256.0
    assert first.budget_phase_tau == 1.0
    assert abs(first.teacher_reliability_q - 0.75) < 1e-12
    assert abs(sum(first.selection_weights.values()) - 1.0) < 1e-12
    assert first.loss_weights["alpha_soft"] > 0.0
    assert first.hidden_dim == 256
    assert first.epochs == 260
    assert first.student_internal_style == "gamlp_like"


def test_t38_teacher_disabled_schedule_zeroes_teacher_terms() -> None:
    schedule = compute_unified_schedule(
        condensed_nodes=512,
        num_classes=64,
        teacher_valid_acc=None,
        majority_valid_acc=None,
        num_nodes=10_000,
        num_teacher_nodes=10_000,
    )

    assert schedule.teacher_reliability_q == 0.0
    assert schedule.selection_weights["soft"] == 0.0
    assert schedule.selection_weights["boundary"] == 0.0
    assert schedule.loss_weights["alpha_soft"] == 0.0
    assert schedule.hidden_dim == 128
    assert schedule.epochs == 220


def test_t38_teacher_cache_policy_uses_byte_budget_not_dataset_name() -> None:
    small = compute_teacher_cache_policy(num_nodes=200_000, num_teacher_nodes=200_000, num_classes=41)
    medium = compute_teacher_cache_policy(num_nodes=2_449_029, num_teacher_nodes=2_449_029, num_classes=47)
    ultra = compute_teacher_cache_policy(num_nodes=111_059_956, num_teacher_nodes=1_546_782, num_classes=172)

    assert small.cache_mode == "dense_fp16"
    assert medium.cache_mode == "dense_fp16"
    assert ultra.cache_mode == "topk8_tail"
    assert ultra.uses_dense_all_node_teacher_cache is False


def test_t38_full_node_ratio_accounting_uses_original_num_nodes() -> None:
    assert full_node_ratio(condensed_nodes=111_060, original_num_nodes=111_059_956) == 111_060 / 111_059_956
