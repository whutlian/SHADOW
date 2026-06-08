from shadow_hgc.shadows.adaptive import adaptive_assignment_b, adaptive_shadow_budgets


def test_adaptive_shadow_allocation_is_deterministic_and_clamped():
    rank = {
        "actor--acts_in-->movie": {"stable_rank": 12.0, "entropy_effective_rank": 8.0},
        "director--directs-->movie": {"stable_rank": 1.5, "entropy_effective_rank": 2.0},
    }

    first = adaptive_shadow_budgets(
        rank,
        effective_M_tau=20,
        shadow_min_per_relation=8,
        shadow_max_multiplier=2.0,
    )
    second = adaptive_shadow_budgets(
        rank,
        effective_M_tau=20,
        shadow_min_per_relation=8,
        shadow_max_multiplier=2.0,
    )

    assert first == second
    assert first["actor--acts_in-->movie"] > first["director--directs-->movie"]
    assert all(8 <= value <= 40 for value in first.values())


def test_adaptive_assignment_b_uses_reconstruction_difficulty():
    assert adaptive_assignment_b(0.20, b_max=4) == 1
    assert adaptive_assignment_b(0.40, b_max=4) == 2
    assert adaptive_assignment_b(0.75, b_max=4) == 4
    assert adaptive_assignment_b(0.75, b_max=2) == 2
