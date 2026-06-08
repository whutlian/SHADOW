import pytest

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.prototype.budgets import (
    allocate_classwise_budgets,
    allocate_shadow_budgets,
    compute_target_budget_from_ratio,
    validate_budget_mode_args,
)


def test_ratio_one_percent_of_ten_thousand_requests_one_hundred():
    result = compute_target_budget_from_ratio(
        num_train_target_nodes=10_000,
        num_train_classes=10,
        ratio=0.01,
        min_proto_per_class=1,
    )

    assert result["requested_target_budget"] == 100
    assert result["effective_target_prototypes"] == 100
    assert result["effective_target_ratio"] == pytest.approx(0.01)


def test_ratio_budget_is_upshifted_by_minimum_class_requirement():
    result = compute_target_budget_from_ratio(
        num_train_target_nodes=1_000,
        num_train_classes=40,
        ratio=0.01,
        min_proto_per_class=4,
    )

    assert result["requested_target_budget"] == 10
    assert result["min_required_target_budget"] == 160
    assert result["effective_target_prototypes"] == 160
    assert result["budget_upshifted"] is True


def test_classwise_budgets_match_effective_budget_when_feasible():
    budgets = allocate_classwise_budgets(
        class_counts={0: 100, 1: 25, 2: 25},
        target_budget=12,
        min_proto_per_class=2,
        exponent=0.5,
    )

    assert sum(budgets.values()) == 12
    assert all(value >= 2 for value in budgets.values())
    assert set(budgets) == {0, 1, 2}


def test_classwise_budgets_keep_minimum_when_requested_budget_is_too_small():
    budgets = allocate_classwise_budgets(
        class_counts={0: 100, 1: 25, 2: 25},
        target_budget=3,
        min_proto_per_class=2,
        exponent=0.5,
    )

    assert sum(budgets.values()) == 6
    assert all(value == 2 for value in budgets.values())


def test_shadow_budgets_use_effective_target_prototypes():
    relations = [
        DirectedRelation("paper", "cite_ref", "paper"),
        DirectedRelation("paper", "cited_by", "paper"),
        DirectedRelation("author", "writes", "paper"),
    ]

    budgets = allocate_shadow_budgets(
        effective_target_prototypes=160,
        relations=relations,
        target_type="paper",
        shadow_ratio_target_target=0.5,
        shadow_ratio_non_target=1.0,
        min_shadow_per_relation=8,
    )

    assert budgets["paper--cite_ref-->paper"] == 40
    assert budgets["paper--cited_by-->paper"] == 40
    assert budgets["author--writes-->paper"] == 160


def test_ratio_and_count_budget_arguments_conflict():
    with pytest.raises(ValueError, match="provide either ratio or target_budget"):
        validate_budget_mode_args(budget_mode="ratio", ratio=0.01, target_budget=32)

    with pytest.raises(ValueError, match="ratio budget mode requires ratio"):
        validate_budget_mode_args(budget_mode="ratio", ratio=None, target_budget=None)

    with pytest.raises(ValueError, match="count budget mode requires target_budget"):
        validate_budget_mode_args(budget_mode="count", ratio=None, target_budget=None)
