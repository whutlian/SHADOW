from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_selector import compute_selected_soft_prior, soft_prior_kl


def score_ttc_rows_for_progressive_compression(
    probs: torch.Tensor,
    *,
    teacher_prior: torch.Tensor,
    entropy: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
) -> torch.Tensor:
    p = probs.detach().float().clamp_min(1e-12)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    prior = teacher_prior.detach().float().clamp_min(1e-12)
    prior = prior / prior.sum().clamp_min(1e-12)
    confidence = p.max(dim=1).values
    cls = p.argmax(dim=1)
    rarity = 1.0 / prior[cls].clamp_min(1e-6)
    if entropy is None:
        entropy = -(p * p.log()).sum(dim=1)
    if margin is None:
        top2 = torch.topk(p, k=min(2, p.shape[1]), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    boundary = 1.0 - margin.detach().float().clamp(0.0, 1.0)
    entropy_term = entropy.detach().float()
    entropy_term = entropy_term / entropy_term.max().clamp_min(1e-12)
    return 0.50 * confidence + 0.25 * rarity / rarity.max().clamp_min(1e-12) + 0.15 * boundary + 0.10 * entropy_term


def select_progressive_subset(scores: torch.Tensor, target_budget: int) -> torch.Tensor:
    if target_budget <= 0:
        raise ValueError("target_budget must be positive")
    count = min(int(target_budget), int(scores.numel()))
    return torch.argsort(scores.detach().float(), descending=True)[:count].to(torch.long)


def add_allnode_repair_rows(
    base_selected: torch.Tensor,
    source_probs: torch.Tensor,
    all_probs: torch.Tensor,
    *,
    teacher_prior: torch.Tensor,
    target_budget: int,
    repair_fraction: float,
) -> torch.Tensor:
    del source_probs
    selected = [int(v) for v in base_selected.detach().cpu().tolist()]
    selected = selected[: int(target_budget)]
    seen = set(selected)
    probs = all_probs.detach().float().cpu()
    prior = teacher_prior.detach().float().cpu()
    repair_slots = max(0, int(round(float(target_budget) * float(repair_fraction))))
    while len(selected) < int(target_budget) and repair_slots > 0:
        current = compute_selected_soft_prior(probs[torch.tensor(selected, dtype=torch.long)]) if selected else torch.zeros_like(prior)
        target_class = int(torch.argmax(prior - current).item())
        scores = probs[:, target_class].clone()
        if seen:
            scores[torch.tensor(sorted(seen), dtype=torch.long)] = -1.0
        best_idx = int(torch.argmax(scores).item())
        if best_idx in seen or float(scores[best_idx].item()) < 0.0:
            break
        selected.append(best_idx)
        seen.add(best_idx)
        repair_slots -= 1
    if len(selected) < int(target_budget):
        scores = score_ttc_rows_for_progressive_compression(probs, teacher_prior=prior)
        for idx in torch.argsort(scores, descending=True).tolist():
            if int(idx) in seen:
                continue
            selected.append(int(idx))
            seen.add(int(idx))
            if len(selected) >= int(target_budget):
                break
    return torch.tensor(selected[: int(target_budget)], dtype=torch.long)
