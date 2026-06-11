from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BonsaiSketch:
    sketch: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BonsaiSelection:
    selected_idx: torch.Tensor
    diagnostics: dict[str, Any]


def _standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)


def build_bonsai_sketch(
    blocks: dict[str, torch.Tensor],
    *,
    labels: torch.Tensor,
    degree: torch.Tensor | None = None,
    output_dim: int = 64,
    seed: int = 42,
) -> BonsaiSketch:
    pieces: list[torch.Tensor] = []
    for name in sorted(blocks):
        value = blocks[name]
        if value.ndim != 2:
            raise ValueError("SFT blocks must be rank-2 tensors")
        pieces.append(_standardize(value))
    y = labels.to(torch.float32).view(-1, 1)
    pieces.append(_standardize(y))
    if degree is not None:
        pieces.append(_standardize(torch.log1p(degree.to(torch.float32)).view(-1, 1)))
    full = torch.cat(pieces, dim=1)
    dim = min(max(1, int(output_dim)), int(full.shape[1]))
    if full.shape[1] == dim:
        sketch = full
    else:
        generator = torch.Generator().manual_seed(int(seed))
        proj = torch.randn(full.shape[1], dim, generator=generator) / math.sqrt(float(dim))
        sketch = full @ proj
    return BonsaiSketch(
        sketch=sketch,
        diagnostics={
            "bonsai_input_dim": int(full.shape[1]),
            "bonsai_sketch_dim": int(sketch.shape[1]),
            "uses_full_pairwise": False,
            "uses_exact_pairwise": False,
        },
    )


def _bucketize(sketch: torch.Tensor, lsh_buckets: int) -> torch.Tensor:
    first = sketch[:, 0].contiguous()
    second = sketch[:, 1].contiguous() if sketch.shape[1] > 1 else first
    q = max(2, int(math.sqrt(max(2, int(lsh_buckets)))))
    qs = torch.linspace(0.0, 1.0, q + 1)[1:-1]
    b1 = torch.bucketize(first, torch.quantile(first, qs).contiguous())
    b2 = torch.bucketize(second, torch.quantile(second, qs).contiguous())
    return b1 * q + b2


def lsh_bonsai_select(
    sketch: torch.Tensor,
    labels: torch.Tensor,
    *,
    total_budget: int,
    lsh_buckets: int = 256,
    seed: int = 42,
) -> BonsaiSelection:
    x = sketch.to(torch.float32).cpu()
    y = labels.to(torch.long).cpu()
    budget = min(max(1, int(total_budget)), int(x.shape[0]))
    buckets = _bucketize(x, int(lsh_buckets))
    selected: list[int] = []
    classes = torch.unique(y, sorted=True).tolist()
    if budget >= len(classes):
        for cls in classes:
            pos = torch.nonzero(y == int(cls), as_tuple=False).view(-1)
            selected.append(int(pos[0].item()))
    remaining = budget - len(set(selected))
    bucket_ids = torch.unique(buckets, sorted=True).tolist()
    generator = torch.Generator().manual_seed(int(seed))
    cursor = 0
    while remaining > 0 and bucket_ids:
        bucket = int(bucket_ids[cursor % len(bucket_ids)])
        pos = torch.nonzero(buckets == bucket, as_tuple=False).view(-1)
        pos = pos[torch.randperm(pos.numel(), generator=generator)] if pos.numel() else pos
        for value in pos.tolist():
            if int(value) not in selected:
                selected.append(int(value))
                remaining -= 1
                break
        cursor += 1
        if cursor > len(bucket_ids) * (budget + 1):
            break
    if len(set(selected)) < budget:
        rest = [idx for idx in range(int(x.shape[0])) if idx not in set(selected)]
        order = torch.tensor(rest, dtype=torch.long)
        if order.numel():
            order = order[torch.randperm(order.numel(), generator=generator)]
            selected.extend(int(v) for v in order[: budget - len(set(selected))].tolist())
    unique: list[int] = []
    seen: set[int] = set()
    for idx in selected:
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
        if len(unique) == budget:
            break
    out = torch.tensor(unique, dtype=torch.long)
    return BonsaiSelection(
        selected_idx=out,
        diagnostics={
            "bonsai_lsh_buckets": int(lsh_buckets),
            "bonsai_selected": int(out.numel()),
            "uses_full_pairwise": False,
            "uses_exact_pairwise": False,
            "class_floor_respected": bool(budget >= len(classes)),
        },
    )
