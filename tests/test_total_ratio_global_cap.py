from shadow_hgc.shadows.adaptive import adaptive_shadow_budgets_global_cap


def test_global_cap_never_exceeded():
    rank = {
        "a--r-->p": {"stable_rank": 100.0},
        "b--r-->p": {"stable_rank": 80.0},
        "c--r-->p": {"stable_rank": 60.0},
    }

    budgets = adaptive_shadow_budgets_global_cap(
        rank,
        effective_M_tau=50,
        total_shadow_budget=17,
        shadow_min_per_relation=4,
        beta=2.0,
    )

    assert sum(budgets.values()) <= 17
    assert set(budgets) == set(rank)
    assert all(value >= 4 for value in budgets.values())


def test_global_cap_handles_budget_below_all_relation_minimums():
    rank = {
        "a--r-->p": {"stable_rank": 9.0},
        "b--r-->p": {"stable_rank": 4.0},
        "c--r-->p": {"stable_rank": 1.0},
    }

    budgets = adaptive_shadow_budgets_global_cap(
        rank,
        effective_M_tau=20,
        total_shadow_budget=7,
        shadow_min_per_relation=3,
    )

    assert sum(budgets.values()) <= 7
    assert all(value >= 0 for value in budgets.values())
    assert budgets["a--r-->p"] >= budgets["b--r-->p"] >= budgets["c--r-->p"]
