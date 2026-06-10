from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class SFTHerdingResult:
    selected_rows: torch.Tensor
    labels: torch.Tensor
    budget_by_class: dict[int, int]
    diagnostics: dict[str, object]


def sqrt_class_budget(labels: torch.Tensor, train_rows: torch.Tensor, total_budget: int) -> dict[int, int]:
    y = labels[train_rows].to(torch.long)
    classes = torch.unique(y, sorted=True)
    if classes.numel() == 0:
        return {}
    counts = {int(c.item()): int((y == c).sum().item()) for c in classes}
    weights = {c: counts[c] ** 0.5 for c in counts}
    denom = sum(weights.values()) or 1.0
    raw = {c: max(1, int(round(int(total_budget) * weights[c] / denom))) for c in counts}
    while sum(raw.values()) > int(total_budget) and any(v > 1 for v in raw.values()):
        c = max(raw, key=lambda key: raw[key])
        raw[c] -= 1
    while sum(raw.values()) < int(total_budget):
        c = max(raw, key=lambda key: counts[key] / max(1, raw[key]))
        raw[c] += 1
    return raw


def _nearest_to_centroid(features: torch.Tensor, rows: torch.Tensor, k: int) -> torch.Tensor:
    values = features[rows].to(torch.float32)
    centroid = values.mean(dim=0, keepdim=True)
    dist = ((values - centroid) ** 2).sum(dim=1)
    order = torch.argsort(dist)
    return rows[order[: int(k)]]


def _herding(features: torch.Tensor, rows: torch.Tensor, k: int) -> torch.Tensor:
    values = features[rows].to(torch.float32)
    centroid = values.mean(dim=0)
    selected: list[int] = []
    running = torch.zeros_like(centroid)
    available = torch.ones(rows.numel(), dtype=torch.bool)
    for step in range(int(k)):
        target = centroid * float(step + 1) - running
        dist = ((values - target.view(1, -1)) ** 2).sum(dim=1)
        dist[~available] = float("inf")
        idx = int(torch.argmin(dist).item())
        selected.append(idx)
        available[idx] = False
        running += values[idx]
    return rows[torch.tensor(selected, dtype=torch.long)]


def select_sft_herding(
    *,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    total_budget: int,
    mode: Literal["medoid", "herding"] = "herding",
    train_losses: torch.Tensor | None = None,
    seed: int = 42,
) -> SFTHerdingResult:
    del seed
    rows = train_rows.to(torch.long)
    budget = sqrt_class_budget(labels, rows, int(total_budget))
    selected: list[torch.Tensor] = []
    y = labels.to(torch.long)
    for cls, cls_budget in budget.items():
        cls_rows = rows[y[rows] == int(cls)]
        if cls_rows.numel() == 0:
            continue
        k = min(int(cls_budget), int(cls_rows.numel()))
        if mode == "medoid":
            chosen = _nearest_to_centroid(signatures, cls_rows, k)
        else:
            chosen = _herding(signatures, cls_rows, k)
        if train_losses is not None and k >= 3:
            losses = train_losses[cls_rows].to(torch.float32)
            loss_order = cls_rows[torch.argsort(losses, descending=True)]
            hard_k = max(1, int(round(k * 0.2)))
            chosen = torch.unique(torch.cat([chosen[: max(0, k - hard_k)], loss_order[:hard_k]]), sorted=False)[:k]
        selected.append(chosen)
    out = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
    return SFTHerdingResult(
        selected_rows=out,
        labels=y[out],
        budget_by_class=budget,
        diagnostics={
            "mode": mode,
            "classwise_budget": True,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "selected_count": int(out.numel()),
        },
    )
