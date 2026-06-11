from __future__ import annotations

import json
import math
from typing import Any

import torch

from shadow_hgc.sft.teacher_transport import (
    TTCCondensedTable,
    _as_probabilities,
    _coverage,
    _rare_structure_scores,
    _round_robin_by_class,
    _take_unique,
    teacher_probability_diagnostics,
)


def _allocate(num_rows: int, weights: dict[str, float]) -> dict[str, int]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    total = float(sum(max(0.0, v) for v in weights.values()))
    if total <= 0.0:
        return {"core": int(num_rows)}
    normalized = {k: max(0.0, float(v)) / total for k, v in weights.items()}
    raw = {k: normalized[k] * int(num_rows) for k in normalized}
    alloc = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = int(num_rows) - sum(alloc.values())
    order = sorted(raw, key=lambda k: (raw[k] - alloc[k], normalized[k]), reverse=True)
    for key in order[:remaining]:
        alloc[key] += 1
    return {k: v for k, v in alloc.items() if v > 0}


def build_ratio_adaptive_budget(num_rows: int, ratio: float, policy: str) -> dict[str, int]:
    policy = str(policy)
    ratio = float(ratio)
    if "core70" in policy:
        weights = {"core": 0.70, "boundary": 0.12, "disagreement": 0.08, "rare_structure": 0.05, "prior_repair": 0.05}
    elif "core55" in policy:
        weights = {"core": 0.55, "boundary": 0.18, "disagreement": 0.12, "rare_structure": 0.08, "prior_repair": 0.07}
    elif "core40" in policy:
        weights = {"core": 0.40, "boundary": 0.25, "disagreement": 0.20, "rare_structure": 0.08, "prior_repair": 0.07}
    elif "confidence" in policy:
        weights = {"core": 0.76, "boundary": 0.08, "disagreement": 0.06, "rare_structure": 0.05, "prior_repair": 0.05}
    elif "disagreement" in policy:
        weights = {"core": 0.45, "boundary": 0.15, "disagreement": 0.28, "rare_structure": 0.05, "prior_repair": 0.07}
    elif "coverage_boundary" in policy:
        weights = {"core": 0.45, "boundary": 0.24, "disagreement": 0.16, "rare_structure": 0.08, "prior_repair": 0.07}
    else:
        weights = {"core": 0.50, "boundary": 0.20, "disagreement": 0.15, "rare_structure": 0.08, "prior_repair": 0.07}
    if ratio >= 0.0025 and ("mixup" in policy or "calibrated_mixup" in policy):
        weights = {k: v * 0.92 for k, v in weights.items()}
        weights["mixup"] = 0.08
    if ratio <= 0.001:
        weights.pop("mixup", None)
    return _allocate(int(num_rows), weights)


def compute_selected_soft_prior(selected_probs: torch.Tensor) -> torch.Tensor:
    probs = _as_probabilities(selected_probs)
    if probs.numel() == 0:
        raise ValueError("selected_probs cannot be empty")
    prior = probs.mean(dim=0)
    return prior / prior.sum().clamp_min(1e-12)


def soft_prior_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    pp = p.detach().float().clamp_min(1e-12)
    qq = q.detach().float().clamp_min(1e-12)
    pp = pp / pp.sum().clamp_min(1e-12)
    qq = qq / qq.sum().clamp_min(1e-12)
    return float((pp * (pp.log() - qq.log())).sum().item())


def repair_soft_class_prior(selected: torch.Tensor, probs: torch.Tensor, teacher_prior: torch.Tensor, budget: int) -> torch.Tensor:
    selected_ids = [int(v) for v in selected.detach().cpu().tolist()]
    seen = set(selected_ids)
    all_probs = _as_probabilities(probs).cpu()
    target = teacher_prior.detach().float().cpu()
    target = target / target.sum().clamp_min(1e-12)
    while len(selected_ids) < int(budget):
        if selected_ids:
            current = compute_selected_soft_prior(all_probs[torch.tensor(selected_ids, dtype=torch.long)])
        else:
            current = torch.zeros_like(target)
        deficit = target - current
        target_class = int(torch.argmax(deficit).item())
        scores = all_probs[:, target_class].clone()
        if seen:
            scores[torch.tensor(sorted(seen), dtype=torch.long)] = -1.0
        best_idx = int(torch.argmax(scores).item())
        if best_idx in seen or float(scores[best_idx].item()) < 0.0:
            break
        selected_ids.append(best_idx)
        seen.add(best_idx)
    return torch.tensor(selected_ids[: int(budget)], dtype=torch.long)


def _bucket_coverage(scores: torch.Tensor, selected: torch.Tensor, buckets: int = 5) -> float:
    if scores.numel() == 0 or selected.numel() == 0:
        return 0.0
    quantiles = torch.linspace(0.0, 1.0, int(buckets) + 1)[1:-1]
    cuts = torch.quantile(scores.detach().float(), quantiles)
    bucket = torch.bucketize(scores.detach().float(), cuts)
    return float(bucket[selected].unique().numel() / max(1, int(buckets)))


def _class_coverage_stats(y_soft: torch.Tensor) -> tuple[int, float, int]:
    pred = y_soft.argmax(dim=1)
    counts = torch.bincount(pred, minlength=y_soft.shape[1])
    positive = counts[counts > 0].float()
    if positive.numel() == 0:
        return 0, 0.0, 0
    return int(positive.min().item()), float(positive.median().item()), int(positive.max().item())


def _counts_json(types: list[str]) -> str:
    counts: dict[str, int] = {}
    for row_type in types:
        counts[row_type] = counts.get(row_type, 0) + 1
    return json.dumps(counts, sort_keys=True)


def select_ttc_rows_ratio_adaptive(
    *,
    features: torch.Tensor,
    teacher_probs: torch.Tensor,
    labels: torch.Tensor | None,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor | None = None,
    test_idx: torch.Tensor | None = None,
    num_rows: int,
    ratio: float,
    policy: str,
    seed: int = 42,
    disagreement: torch.Tensor | None = None,
    mixup_alpha: float = 0.4,
) -> TTCCondensedTable:
    del valid_idx, test_idx
    x = features.detach().float().cpu()
    probs = _as_probabilities(teacher_probs).cpu()
    if x.shape[0] != probs.shape[0]:
        raise ValueError("features and teacher_probs must have the same first dimension")
    n, d = x.shape
    allocation = build_ratio_adaptive_budget(int(num_rows), ratio, policy)
    confidence = probs.max(dim=1).values
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    pred = probs.argmax(dim=1)
    dis = disagreement.detach().float().cpu() if disagreement is not None else entropy
    rare = _rare_structure_scores(x)

    used: set[int] = set()
    selected_ids: list[int] = []
    row_types: list[str] = []

    def add(row_type: str, ids: list[int]) -> None:
        selected_ids.extend(ids)
        row_types.extend([row_type] * len(ids))

    add("core", _round_robin_by_class(confidence, pred, allocation.get("core", 0), used, largest=True))
    add("boundary", _round_robin_by_class(margin, pred, allocation.get("boundary", 0), used, largest=False))
    add("disagreement", _take_unique(dis, allocation.get("disagreement", 0), used, largest=True))
    add("rare_structure", _take_unique(rare, allocation.get("rare_structure", 0), used, largest=True))

    real_budget = int(num_rows) - allocation.get("mixup", 0)
    if len(selected_ids) < real_budget:
        repaired = repair_soft_class_prior(torch.tensor(selected_ids, dtype=torch.long), probs, probs.mean(dim=0), real_budget)
        for idx in repaired.tolist()[len(selected_ids) :]:
            if int(idx) not in used:
                used.add(int(idx))
            selected_ids.append(int(idx))
            row_types.append("prior_repair")
    if len(selected_ids) < real_budget:
        add("core", _take_unique(confidence, real_budget - len(selected_ids), used, largest=True))

    selected_ids = selected_ids[:real_budget]
    row_types = row_types[:real_budget]
    z_rows = [x[idx] for idx in selected_ids]
    y_rows = [probs[idx] for idx in selected_ids]
    source_ids = selected_ids[:]

    mix_count = allocation.get("mixup", 0)
    if mix_count > 0:
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        boundary_order = torch.argsort(margin, descending=False).tolist()
        base = selected_ids if selected_ids else list(range(n))
        beta = torch.distributions.Beta(float(mixup_alpha), float(mixup_alpha))
        for i in range(mix_count):
            a = int(base[i % len(base)])
            b = int(boundary_order[(i + int(seed)) % len(boundary_order)])
            lam = float(beta.sample((1,)).item())
            if not torch.isfinite(torch.tensor(lam)):
                lam = 0.5
            z_rows.append(lam * x[a] + (1.0 - lam) * x[b])
            y_rows.append(lam * probs[a] + (1.0 - lam) * probs[b])
            source_ids.append(-1)
            row_types.append("mixup")

    while len(z_rows) < int(num_rows):
        idx = int(torch.argsort(confidence, descending=True)[len(z_rows) % n])
        z_rows.append(x[idx])
        y_rows.append(probs[idx])
        source_ids.append(idx)
        row_types.append("core")

    z_syn = torch.stack(z_rows[: int(num_rows)], dim=0).view(int(num_rows), d)
    y_syn_soft = _as_probabilities(torch.stack(y_rows[: int(num_rows)], dim=0))
    source_node_ids = torch.tensor(source_ids[: int(num_rows)], dtype=torch.long)
    train_set = {int(v) for v in train_idx.detach().cpu().tolist()}
    hard_anchor_mask = torch.tensor([int(v) >= 0 and int(v) in train_set for v in source_node_ids.tolist()], dtype=torch.bool)
    y_syn_hard = torch.full((int(num_rows),), -1, dtype=torch.long)
    if labels is not None and hard_anchor_mask.any():
        labels_cpu = labels.detach().cpu().long()
        y_syn_hard[hard_anchor_mask] = labels_cpu[source_node_ids[hard_anchor_mask]]
    selected_real = source_node_ids[source_node_ids >= 0]
    class_min, class_median, class_max = _class_coverage_stats(y_syn_soft)
    diagnostics: dict[str, Any] = {
        **teacher_probability_diagnostics(probs, disagreement=disagreement),
        "condensed_nodes": int(num_rows),
        "candidate_nodes": "all",
        "candidate_node_count": int(n),
        "budget_policy": str(policy),
        "budget_allocation": allocation,
        "row_type_counts_json": _counts_json(row_types[: int(num_rows)]),
        "selected_bucket_counts": {k: int(v) for k, v in allocation.items()},
        "selected_soft_prior_kl": soft_prior_kl(compute_selected_soft_prior(y_syn_soft), probs.mean(dim=0)),
        "entropy_bucket_coverage": _bucket_coverage(entropy, selected_real),
        "margin_bucket_coverage": _bucket_coverage(-margin, selected_real),
        "disagreement_bucket_coverage": _bucket_coverage(dis, selected_real),
        "degree_bucket_coverage": _coverage(rare, selected_real),
        "class_coverage_min": class_min,
        "class_coverage_median": class_median,
        "class_coverage_max": class_max,
        "hard_anchor_count": int(hard_anchor_mask.sum().item()),
        "soft_only_count": int((~hard_anchor_mask).sum().item()),
        "mixup_virtual_count": int(sum(1 for t in row_types[: int(num_rows)] if t == "mixup")),
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
    }
    return TTCCondensedTable(
        z_syn=z_syn,
        y_syn_soft=y_syn_soft,
        y_syn_hard=y_syn_hard,
        hard_anchor_mask=hard_anchor_mask,
        source_node_ids=source_node_ids,
        bucket_types=row_types[: int(num_rows)],
        sample_weight=torch.ones(int(num_rows), dtype=torch.float32),
        diagnostics=diagnostics,
    )
