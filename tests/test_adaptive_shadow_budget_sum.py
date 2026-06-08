from shadow_hgc.shadows.adaptive import adaptive_shadow_budgets_global_cap


def test_global_cap_preserves_minimums_when_possible_and_distributes_remainder_by_rank():
    rank = {
        "low--r-->paper": {"stable_rank": 1.0, "entropy_effective_rank": 1.0},
        "mid--r-->paper": {"stable_rank": 4.0, "entropy_effective_rank": 2.0},
        "high--r-->paper": {"stable_rank": 9.0, "entropy_effective_rank": 3.0},
    }

    budgets = adaptive_shadow_budgets_global_cap(
        rank,
        effective_M_tau=30,
        total_shadow_budget=15,
        shadow_min_per_relation=3,
        beta=1.0,
    )

    assert sum(budgets.values()) == 15
    assert all(value >= 3 for value in budgets.values())
    assert budgets["high--r-->paper"] >= budgets["mid--r-->paper"] >= budgets["low--r-->paper"]


def test_global_cap_returns_zero_budgets_for_zero_total_cap():
    budgets = adaptive_shadow_budgets_global_cap(
        {"a--r-->p": {"stable_rank": 5.0}},
        effective_M_tau=10,
        total_shadow_budget=0,
        shadow_min_per_relation=2,
    )

    assert budgets == {"a--r-->p": 0}
