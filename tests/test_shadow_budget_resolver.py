from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.shadows.budgets import resolve_shadow_budgets


def test_shadow_budget_resolver_uses_effective_target_budget_by_relation_type():
    cite = DirectedRelation("paper", "cite_ref", "paper")
    cited = DirectedRelation("paper", "cited_by", "paper")
    writes = DirectedRelation("author", "writes", "paper")

    budgets = resolve_shadow_budgets(
        relations=[cite, cited, writes],
        target_type="paper",
        effective_M_tau=64,
        requested_M_r=None,
        min_shadows_per_relation=8,
    )

    assert budgets[cite] == 16
    assert budgets[cited] == 16
    assert budgets[writes] == 64


def test_shadow_budget_resolver_respects_explicit_relation_dict():
    cite = DirectedRelation("paper", "cite_ref", "paper")
    writes = DirectedRelation("author", "writes", "paper")

    budgets = resolve_shadow_budgets(
        relations=[cite, writes],
        target_type="paper",
        effective_M_tau=64,
        requested_M_r={str(cite): 5, str(writes): 11},
    )

    assert budgets[cite] == 5
    assert budgets[writes] == 11


def test_shadow_budget_resolver_expands_scalar_request_to_all_relations():
    cite = DirectedRelation("paper", "cite_ref", "paper")
    writes = DirectedRelation("author", "writes", "paper")

    budgets = resolve_shadow_budgets(
        relations=[cite, writes],
        target_type="paper",
        effective_M_tau=64,
        requested_M_r=12,
    )

    assert budgets == {cite: 12, writes: 12}
