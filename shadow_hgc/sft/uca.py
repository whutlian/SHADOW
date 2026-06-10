from __future__ import annotations

from typing import Any

import torch


def _domain_centers(signatures: torch.Tensor, target_rows: torch.Tensor, num_domains: int, seed: int) -> torch.Tensor:
    target_sig = signatures[target_rows].to(torch.float32)
    if target_sig.numel() == 0:
        return target_sig
    k = max(1, min(int(num_domains), int(target_sig.shape[0])))
    first = int(seed) % int(target_sig.shape[0])
    chosen = [first]
    min_dist = torch.cdist(target_sig, target_sig[first : first + 1]).view(-1)
    for _ in range(1, k):
        idx = int(torch.argmax(min_dist).item())
        chosen.append(idx)
        min_dist = torch.minimum(min_dist, torch.cdist(target_sig, target_sig[idx : idx + 1]).view(-1))
    return target_sig[torch.tensor(chosen, dtype=torch.long)]


def assign_uca_domains(signatures: torch.Tensor, target_rows: torch.Tensor, *, num_domains: int, seed: int = 42) -> torch.Tensor:
    signatures = signatures.to(torch.float32).cpu()
    target_rows = target_rows.to(torch.long).cpu()
    centers = _domain_centers(signatures, target_rows, int(num_domains), int(seed))
    if centers.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.argmin(torch.cdist(signatures[target_rows].to(torch.float32), centers), dim=1).to(torch.long)


def coverage_gap_metrics(all_hist: torch.Tensor, train_hist: torch.Tensor) -> dict[str, Any]:
    all_hist = all_hist.to(torch.float64).cpu()
    train_hist = train_hist.to(torch.float64).cpu()
    all_dist = all_hist / all_hist.sum().clamp_min(1.0)
    train_dist = train_hist / train_hist.sum().clamp_min(1.0)
    gap = all_dist - train_dist
    return {
        "coverage_gap_l1": float(gap.abs().sum().item()),
        "coverage_gap_l2": float(torch.linalg.vector_norm(gap).item()),
        "domains_total": int(max(all_hist.numel(), train_hist.numel())),
        "domains_without_train_support": int((train_hist <= 0).sum().item()),
        "domains_without_unlabeled_support": int((all_hist <= 0).sum().item()),
    }


def select_uca_labeled_nearest(
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    target_rows: torch.Tensor,
    *,
    budget: int,
    num_domains: int,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del labels
    signatures = signatures.to(torch.float32).cpu()
    train_rows = train_rows.to(torch.long).cpu()
    target_rows = target_rows.to(torch.long).cpu()
    budget = max(1, min(int(budget), int(train_rows.numel())))
    domains = assign_uca_domains(signatures, target_rows, num_domains=int(num_domains), seed=int(seed))
    train_target_pos = {int(row): idx for idx, row in enumerate(target_rows.tolist())}
    train_domain_ids = torch.tensor([int(domains[train_target_pos[int(row)]].item()) for row in train_rows if int(row) in train_target_pos], dtype=torch.long)
    train_rows_supported = torch.tensor([int(row) for row in train_rows if int(row) in train_target_pos], dtype=torch.long)
    if train_rows_supported.numel() == 0:
        train_rows_supported = train_rows
        train_domain_ids = torch.zeros(train_rows.numel(), dtype=torch.long)
    all_hist = torch.bincount(domains, minlength=int(num_domains))
    train_hist = torch.bincount(train_domain_ids, minlength=int(num_domains))
    metrics = coverage_gap_metrics(all_hist, train_hist)
    weights = all_hist.to(torch.float64) / all_hist.sum().clamp_min(1).to(torch.float64)
    raw_quota = weights * int(budget)
    quota = torch.floor(raw_quota).to(torch.long)
    remainder = int(budget) - int(quota.sum().item())
    if remainder > 0:
        order = torch.argsort(raw_quota - quota.to(raw_quota.dtype), descending=True)
        quota[order[:remainder]] += 1
    selected: list[int] = []
    for domain in range(int(num_domains)):
        k = int(quota[domain].item())
        if k <= 0:
            continue
        candidates = train_rows_supported[train_domain_ids == domain]
        if candidates.numel() == 0:
            center = signatures[target_rows[domains == domain]].mean(dim=0, keepdim=True) if torch.any(domains == domain) else signatures[train_rows].mean(dim=0, keepdim=True)
            dist = torch.cdist(signatures[train_rows], center).view(-1)
            candidates = train_rows[torch.argsort(dist)]
        else:
            center = signatures[target_rows[domains == domain]].mean(dim=0, keepdim=True)
            dist = torch.cdist(signatures[candidates], center).view(-1)
            candidates = candidates[torch.argsort(dist)]
        for row in candidates.tolist():
            if row not in selected:
                selected.append(int(row))
            if len(selected) >= int(budget) or sum(1 for item in selected if item in candidates.tolist()) >= k:
                break
    if len(selected) < int(budget):
        center = signatures[target_rows].mean(dim=0, keepdim=True)
        order = train_rows[torch.argsort(torch.cdist(signatures[train_rows], center).view(-1))]
        for row in order.tolist():
            if row not in selected:
                selected.append(int(row))
            if len(selected) >= int(budget):
                break
    selected_tensor = torch.tensor(selected[:budget], dtype=torch.long)
    selected_domains = torch.tensor([int(domains[train_target_pos[int(row)]].item()) for row in selected_tensor if int(row) in train_target_pos], dtype=torch.long)
    selected_hist = torch.bincount(selected_domains, minlength=int(num_domains)) if selected_domains.numel() else torch.zeros(int(num_domains), dtype=torch.long)
    selected_gap = coverage_gap_metrics(all_hist, selected_hist)
    stats: dict[str, Any] = {
        **metrics,
        "selected_coverage_gap_l1": selected_gap["coverage_gap_l1"],
        "selected_coverage_gap_l2": selected_gap["coverage_gap_l2"],
        "domain_hist_all": [int(value) for value in all_hist.tolist()],
        "domain_hist_train": [int(value) for value in train_hist.tolist()],
        "domain_hist_selected": [int(value) for value in selected_hist.tolist()],
        "uca_num_domains": int(num_domains),
        "uca_domain_seed": int(seed),
        "uca_uses_valid_test_labels": False,
    }
    return selected_tensor, stats
