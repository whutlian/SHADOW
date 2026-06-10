import torch

from shadow_hgc.sft.fdm_lite import (
    T25_METHODS,
    allocate_t25_class_budgets,
    assign_shadow_b2,
    build_fdm_lite_plan,
    reduce_sft_signature,
    select_fdm_lite_rows,
)


def test_t25_method_names_are_registered():
    assert set(T25_METHODS) == {
        "sft_hnr_random",
        "sft_hnr_fdm_herding",
        "sft_hnr_fdm_kcenter",
        "sft_hnr_fdm_hybrid",
        "sft_hnr_fdm_shadow_b1",
        "sft_hnr_fdm_shadow_b2",
    }


def test_t25_class_budgets_respect_floors_when_feasible():
    labels = torch.tensor([0] * 38 + [1] * 9 + [2] * 3)
    train_rows = torch.arange(labels.numel())
    budgets = allocate_t25_class_budgets(labels, train_rows, total_budget=12, min_per_class=3)

    assert sum(budgets.values()) == 12
    assert budgets[0] >= 3
    assert budgets[1] >= 3
    assert budgets[2] >= 3


def test_t25_class_budget_does_not_loop_on_saturated_rare_class():
    labels = torch.tensor([0] * 100 + [1])
    train_rows = torch.arange(labels.numel())
    budgets = allocate_t25_class_budgets(labels, train_rows, total_budget=20, min_per_class=1)

    assert sum(budgets.values()) == 20
    assert budgets[1] == 1
    assert budgets[0] == 19


def test_t25_fdm_plan_caps_subclasses_and_candidate_pool():
    signature = torch.randn(60, 16, generator=torch.Generator().manual_seed(1))
    labels = torch.tensor([0] * 30 + [1] * 30)
    train_rows = torch.arange(100, 160)
    weights = torch.linspace(0.1, 2.0, 60)
    plan = build_fdm_lite_plan(
        signature,
        labels,
        train_rows,
        total_budget=10,
        node_weight=weights,
        scale_bucket="ultra",
        fdm_k_min=2,
        fdm_k_max=4,
        candidate_rho=4,
        candidate_max=8,
        seed=7,
    )

    assert plan.signature_dim == 16
    assert plan.num_subclasses <= 8
    assert all(pool.candidate_rows.numel() <= 8 for pool in plan.pools)
    assert all(pool.candidate_rows.numel() <= 4 * max(1, pool.budget) for pool in plan.pools)


def test_t25_selectors_return_real_rows_and_do_not_exceed_budget():
    signature = torch.randn(40, 8, generator=torch.Generator().manual_seed(2))
    labels = torch.tensor([0] * 20 + [1] * 20)
    train_rows = torch.arange(200, 240)
    weights = torch.ones(40)
    strata = ["H+"] * 10 + ["H0"] * 10 + ["H-"] * 20

    for method in ["sft_hnr_random", "sft_hnr_fdm_herding", "sft_hnr_fdm_kcenter", "sft_hnr_fdm_hybrid"]:
        result = select_fdm_lite_rows(
            signature,
            labels,
            train_rows,
            total_budget=8,
            method=method,
            node_weight=weights,
            stratum=strata,
            seed=11,
            candidate_rho=4,
            candidate_max=16,
        )
        assert result.selected_rows.numel() <= 8
        assert set(result.selected_rows.tolist()) <= set(train_rows.tolist())
        assert result.diagnostics["uses_exact_pairwise"] is False


def test_t25_reduce_signature_is_deterministic_and_bounded():
    signature = torch.arange(120, dtype=torch.float32).view(10, 12)
    a = reduce_sft_signature(signature, output_dim=5, seed=42)
    b = reduce_sft_signature(signature, output_dim=5, seed=42)

    assert a.shape == (10, 5)
    assert torch.allclose(a, b)


def test_t25_shadow_b2_assignments_are_nonnegative_and_sparse():
    residual = torch.tensor([[1.0, 0.0], [0.1, 0.9], [-1.0, 0.0]])
    shadows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assignment = assign_shadow_b2(residual, shadows)

    assert assignment.src_shadow.numel() == assignment.dst_proto.numel() == assignment.edge_weight.numel()
    assert torch.all(assignment.edge_weight >= 0)
    for proto in range(residual.shape[0]):
        assert int((assignment.dst_proto == proto).sum().item()) <= 2
