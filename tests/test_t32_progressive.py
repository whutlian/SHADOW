from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_progressive import add_allnode_repair_rows, score_ttc_rows_for_progressive_compression, select_progressive_subset
from shadow_hgc.sft.ttcpp_selector import compute_selected_soft_prior, soft_prior_kl


def test_t32_progressive_subset_respects_final_budget() -> None:
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.55, 0.45]])
    scores = score_ttc_rows_for_progressive_compression(probs, teacher_prior=probs.mean(dim=0), entropy=torch.ones(5), margin=torch.ones(5))
    selected = select_progressive_subset(scores, target_budget=3)
    assert selected.numel() == 3
    assert int(selected.max()) < 5


def test_t32_progressive_repair_keeps_budget_and_does_not_worsen_prior() -> None:
    source_probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    all_probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.05, 0.95]])
    teacher_prior = all_probs.mean(dim=0)
    base = torch.tensor([0, 1, 2])
    repaired = add_allnode_repair_rows(base, source_probs, all_probs, teacher_prior=teacher_prior, target_budget=4, repair_fraction=0.25)
    assert repaired.numel() == 4
    before = soft_prior_kl(compute_selected_soft_prior(all_probs[base]), teacher_prior)
    after = soft_prior_kl(compute_selected_soft_prior(all_probs[repaired]), teacher_prior)
    assert after <= before + 1e-8
