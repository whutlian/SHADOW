from __future__ import annotations

from dataclasses import dataclass

from shadow_hgc.prototype.budgets import validate_budget_mode_args


@dataclass(frozen=True)
class BudgetRunSpec:
    budget_mode: str
    ratio: float | None
    target_budget: int | None
    label: str


def ratio_slug(ratio: float) -> str:
    text = f"{float(ratio):.8f}".rstrip("0").rstrip(".")
    return "r" + text.replace(".", "p")


def count_slug(target_budget: int) -> str:
    return f"count{int(target_budget)}"


def make_budget_run_specs(
    *,
    budget_mode: str | None,
    ratios: list[float] | None,
    target_budgets: list[int] | None,
    legacy_target_budgets: list[int] | None = None,
    default_ratios: list[float] | None = None,
    default_target_budgets: list[int] | None = None,
) -> list[BudgetRunSpec]:
    ratio_values = list(ratios or [])
    count_values = list(target_budgets or [])
    if legacy_target_budgets:
        count_values.extend(int(value) for value in legacy_target_budgets)
    if ratio_values and count_values and budget_mode is None:
        raise ValueError("provide either ratios or target budgets, not both")
    resolved_mode = budget_mode
    if resolved_mode is None:
        if ratio_values:
            resolved_mode = "ratio"
        elif count_values:
            resolved_mode = "count"
        elif default_ratios:
            resolved_mode = "ratio"
            ratio_values = list(default_ratios)
        else:
            resolved_mode = "count"
            count_values = list(default_target_budgets or [])
    if resolved_mode == "ratio":
        if count_values:
            raise ValueError("budget_mode=ratio cannot be used with target budgets")
        if not ratio_values:
            ratio_values = list(default_ratios or [])
        if not ratio_values:
            raise ValueError("ratio mode requires --ratio/--ratios")
        specs = [
            BudgetRunSpec("ratio", float(ratio), None, ratio_slug(float(ratio)))
            for ratio in ratio_values
        ]
    elif resolved_mode == "count":
        if ratio_values:
            raise ValueError("budget_mode=count cannot be used with ratios")
        if not count_values:
            count_values = list(default_target_budgets or [])
        if not count_values:
            raise ValueError("count mode requires --target-budget/--target-budgets")
        specs = [
            BudgetRunSpec("count", None, int(target_budget), count_slug(int(target_budget)))
            for target_budget in count_values
        ]
    else:
        raise ValueError("budget_mode must be ratio, count, or omitted")
    for spec in specs:
        validate_budget_mode_args(
            budget_mode=spec.budget_mode,
            ratio=spec.ratio,
            target_budget=spec.target_budget,
        )
    return specs
