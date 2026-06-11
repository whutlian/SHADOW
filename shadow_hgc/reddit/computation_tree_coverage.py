from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CTCSelection:
    selected_pos: torch.Tensor
    diagnostics: dict[str, Any]


def reduce_ctc_sketch(signature: torch.Tensor, *, output_dim: int = 64, seed: int = 42) -> torch.Tensor:
    x = signature.detach().to(torch.float32).cpu()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    output_dim = min(max(1, int(output_dim)), int(x.shape[1]))
    if x.shape[1] == output_dim:
        return x
    generator = torch.Generator().manual_seed(int(seed))
    proj = torch.randn(x.shape[1], output_dim, generator=generator) / math.sqrt(float(output_dim))
    return x @ proj


def _degree_bucket(degree: torch.Tensor) -> torch.Tensor:
    d = degree.detach().to(torch.float32).cpu()
    boundaries = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64], dtype=torch.float32)
    return torch.bucketize(d, boundaries)


def _bucket_ids(sketch: torch.Tensor, labels: torch.Tensor, degree: torch.Tensor | None) -> list[tuple[int, int, int]]:
    y = labels.detach().to(torch.long).cpu()
    first = sketch[:, 0]
    second = sketch[:, 1] if sketch.shape[1] > 1 else sketch[:, 0]
    q1 = torch.bucketize(first, torch.quantile(first, torch.tensor([0.25, 0.50, 0.75])))
    q2 = torch.bucketize(second, torch.quantile(second, torch.tensor([0.25, 0.50, 0.75])))
    deg = _degree_bucket(degree) if degree is not None else torch.zeros_like(q1)
    return [(int(y[i].item()), int(q1[i].item() * 4 + q2[i].item()), int(deg[i].item())) for i in range(y.numel())]


def _select_diverse_from_bucket(pos: torch.Tensor, sketch: torch.Tensor, budget: int, *, seed: int) -> list[int]:
    if budget <= 0 or pos.numel() == 0:
        return []
    if pos.numel() <= budget:
        return [int(v) for v in pos.tolist()]
    sub = sketch[pos]
    center = sub.mean(dim=0, keepdim=True)
    medoid_local = torch.argmin(torch.norm(sub - center, dim=1)).item()
    selected = [int(pos[medoid_local].item())]
    generator = torch.Generator().manual_seed(int(seed))
    perm = pos[torch.randperm(pos.numel(), generator=generator)]
    for value in perm.tolist():
        if len(selected) >= budget:
            break
        if int(value) not in selected:
            selected.append(int(value))
    return selected


def ctc_bucket_selection(
    signature: torch.Tensor,
    labels: torch.Tensor,
    *,
    total_budget: int,
    degree: torch.Tensor | None = None,
    output_dim: int = 64,
    seed: int = 42,
    rare_bucket_floor: int = 1,
) -> CTCSelection:
    x = reduce_ctc_sketch(signature, output_dim=output_dim, seed=seed)
    y = labels.detach().to(torch.long).cpu()
    budget = min(max(1, int(total_budget)), int(y.numel()))
    bucket_keys = _bucket_ids(x, y, degree)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(bucket_keys):
        buckets.setdefault(key, []).append(idx)

    classes = torch.unique(y, sorted=True).tolist()
    selected: list[int] = []
    if budget >= len(classes):
        for cls in classes:
            cls_pos = torch.nonzero(y == int(cls), as_tuple=False).view(-1)
            chosen = _select_diverse_from_bucket(cls_pos, x, 1, seed=int(seed) + int(cls))
            selected.extend(chosen)

    remaining = budget - len(set(selected))
    scores: list[tuple[float, tuple[int, int, int], torch.Tensor]] = []
    for key, values in buckets.items():
        pos = torch.tensor(values, dtype=torch.long)
        mass = len(values)
        degree_weight = 1.0
        if degree is not None:
            degree_weight = float(torch.log1p(degree[pos].to(torch.float32)).mean().item() + 1.0)
        score = 0.50 * math.sqrt(float(mass)) + 0.25 * float(rare_bucket_floor) + 0.15 * degree_weight + 0.10
        scores.append((score, key, pos))
    scores.sort(key=lambda item: (-item[0], item[1]))

    cursor = 0
    while remaining > 0 and scores:
        _, key, pos = scores[cursor % len(scores)]
        available = torch.tensor([int(v) for v in pos.tolist() if int(v) not in set(selected)], dtype=torch.long)
        if available.numel():
            selected.extend(_select_diverse_from_bucket(available, x, 1, seed=int(seed) + cursor + key[0] * 17))
            remaining -= 1
        cursor += 1
        if cursor > len(scores) * (budget + 1):
            break

    if len(set(selected)) < budget:
        selected_set = set(selected)
        rest = [idx for idx in range(y.numel()) if idx not in selected_set]
        generator = torch.Generator().manual_seed(int(seed) + 999)
        order = torch.tensor(rest, dtype=torch.long)
        if order.numel():
            order = order[torch.randperm(order.numel(), generator=generator)]
            selected.extend(int(v) for v in order[: budget - len(set(selected))].tolist())

    unique = []
    seen = set()
    for value in selected:
        if value not in seen:
            unique.append(value)
            seen.add(value)
        if len(unique) == budget:
            break
    out = torch.tensor(unique, dtype=torch.long)
    return CTCSelection(
        selected_pos=out,
        diagnostics={
            "ctc_num_buckets": len(buckets),
            "ctc_sketch_dim": int(x.shape[1]),
            "ctc_total_budget": int(budget),
            "ctc_selected": int(out.numel()),
            "class_floor_respected": bool(budget >= len(classes)),
            "rare_bucket_floor": int(rare_bucket_floor),
            "runs_full_all_pair_original_search": False,
        },
    )
