import pytest

from shadow_hgc.eval.budgeting import make_budget_run_specs, ratio_slug


def test_ratio_slug_is_filename_safe():
    assert ratio_slug(0.01) == "r0p01"
    assert ratio_slug(0.0025) == "r0p0025"
    assert ratio_slug(0.0001) == "r0p0001"


def test_make_budget_run_specs_prefers_ratio_when_ratios_are_given():
    specs = make_budget_run_specs(
        budget_mode=None,
        ratios=[0.005, 0.01],
        target_budgets=None,
    )

    assert [(spec.budget_mode, spec.ratio, spec.target_budget, spec.label) for spec in specs] == [
        ("ratio", 0.005, None, "r0p005"),
        ("ratio", 0.01, None, "r0p01"),
    ]


def test_make_budget_run_specs_uses_count_labels_for_target_budgets():
    specs = make_budget_run_specs(
        budget_mode=None,
        ratios=None,
        target_budgets=[32],
        legacy_target_budgets=[64],
    )

    assert [(spec.budget_mode, spec.ratio, spec.target_budget, spec.label) for spec in specs] == [
        ("count", None, 32, "count32"),
        ("count", None, 64, "count64"),
    ]


def test_make_budget_run_specs_rejects_ambiguous_ratio_and_count_inputs():
    with pytest.raises(ValueError, match="provide either ratios or target budgets"):
        make_budget_run_specs(
            budget_mode=None,
            ratios=[0.01],
            target_budgets=[32],
        )

    with pytest.raises(ValueError, match="budget_mode=ratio cannot be used"):
        make_budget_run_specs(
            budget_mode="ratio",
            ratios=[0.01],
            target_budgets=[32],
        )
