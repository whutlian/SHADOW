from __future__ import annotations

from shadow_hgc.sft.t40_contract import FIXED_CANDIDATE_POLICIES
from shadow_hgc.sft.unified_auto_v2 import (
    apply_candidate_policy,
    compute_t40_schedule,
    compute_teacher_cache_policy_v2,
    policy_selection_score,
    select_best_candidate,
)


def test_t40_teacher_cache_policy_prefers_dense_for_reddit_when_budget_fits() -> None:
    decision = compute_teacher_cache_policy_v2(
        num_nodes=232_965,
        num_teacher_nodes=232_965,
        num_classes=41,
        is_ultra_dataset=False,
        dense_budget_bytes=256 * 1024 * 1024,
    )

    assert decision.policy == "auto_by_bytes"
    assert decision.cache_mode == "dense_fp16"
    assert decision.cache_k == 0
    assert decision.uses_dense_all_node_teacher_cache is False


def test_t40_teacher_cache_policy_forces_topk_for_papers100m() -> None:
    decision = compute_teacher_cache_policy_v2(
        num_nodes=111_059_956,
        num_teacher_nodes=1_546_782,
        num_classes=172,
        is_ultra_dataset=True,
        dense_budget_bytes=10 * 1024 * 1024 * 1024,
    )

    assert decision.policy == "auto_by_bytes"
    assert decision.cache_mode in {"topk8_tail", "topk16_tail"}
    assert decision.cache_k in {8, 16}
    assert decision.uses_dense_all_node_teacher_cache is False


def test_t40_schedule_v2_adds_domain_weight_and_capacity_rules() -> None:
    low_gap = compute_t40_schedule(
        condensed_nodes=800,
        num_classes=40,
        teacher_valid_acc=None,
        majority_valid_acc=0.14,
        domain_gap_train_all=0.02,
        num_nodes=169_343,
    )
    high_gap = compute_t40_schedule(
        condensed_nodes=12_245,
        num_classes=47,
        teacher_valid_acc=0.90,
        majority_valid_acc=0.095,
        domain_gap_train_all=0.28,
        num_nodes=2_449_029,
    )

    assert low_gap.class_capacity_b == 20.0
    assert low_gap.hidden_dim == 128
    assert low_gap.epochs == 220
    assert high_gap.selection_weights["domain"] > low_gap.selection_weights["domain"]
    assert high_gap.loss_weights["lambda_domain"] > low_gap.loss_weights["lambda_domain"]
    assert high_gap.student_internal_style == "gamlp_like"
    assert high_gap.hidden_dim == 256
    assert high_gap.epochs == 300


def test_t40_candidate_policy_modifiers_keep_single_public_family() -> None:
    base = compute_t40_schedule(
        condensed_nodes=1165,
        num_classes=41,
        teacher_valid_acc=0.94,
        majority_valid_acc=0.31,
        domain_gap_train_all=0.05,
        num_nodes=232_965,
    )

    candidates = [apply_candidate_policy(base, policy) for policy in FIXED_CANDIDATE_POLICIES]

    assert [candidate.policy_name for candidate in candidates] == list(FIXED_CANDIDATE_POLICIES)
    assert all(candidate.student_family == "STT-GatedMixer" for candidate in candidates)
    assert candidates[1].selection_weights["coverage"] > candidates[0].selection_weights["coverage"]
    assert candidates[1].selection_weights["domain"] == 0.0
    assert candidates[2].selection_weights["domain"] > candidates[0].selection_weights["domain"]
    assert candidates[3].selection_weights["soft"] > candidates[0].selection_weights["soft"]
    assert candidates[4].hidden_dim >= candidates[0].hidden_dim


def test_t40_candidate_selection_uses_composite_valid_score() -> None:
    rows = [
        {"selected_policy": "auto_base", "valid_acc": 0.90, "valid_macro_f1": 0.50, "selected_prior_kl": 0.0, "domain_coverage_gap": 0.40},
        {"selected_policy": "domain_coverage", "valid_acc": 0.899, "valid_macro_f1": 0.50, "selected_prior_kl": 0.0, "domain_coverage_gap": 0.05},
        {"selected_policy": "teacher_transport", "valid_acc": 0.901, "valid_macro_f1": 0.30, "selected_prior_kl": 0.30, "domain_coverage_gap": 0.40},
    ]

    scored = [policy_selection_score(row) for row in rows]
    best = select_best_candidate(rows)

    assert scored[1] > scored[0]
    assert best["selected_policy"] == "domain_coverage"
    assert best["policy_selection_score"] == scored[1]
