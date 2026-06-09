from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class CorrectSmoothResult:
    logits: torch.Tensor
    diagnostics: dict[str, Any]


def select_best_validation_row(rows: list[dict[str, Any]], *, metric: str = "valid_acc", tie_break: str = "valid_macro_f1") -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")

    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (float(row.get(metric, 0.0) or 0.0), float(row.get(tie_break, 0.0) or 0.0))

    return max(rows, key=key)


def _propagate(values: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, steps: int) -> torch.Tensor:
    z = values.to(torch.float32)
    if int(steps) <= 0:
        return z
    edge_index = edge_index.to(device=z.device, dtype=torch.long)
    alpha = destination_row_normalize(edge_index, int(num_nodes)).to(device=z.device, dtype=torch.float32)
    for _ in range(int(steps)):
        out = torch.zeros(int(num_nodes), int(z.shape[1]), dtype=torch.float32, device=z.device)
        if edge_index.numel() > 0:
            out.index_add_(0, edge_index[1], z[edge_index[0]] * alpha.unsqueeze(1))
        z = out
    return z


def smooth_probabilities(
    *,
    probabilities: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float,
    steps: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    p0 = probabilities.to(torch.float32)
    if int(steps) <= 0 or float(alpha) == 0.0:
        return p0 / p0.sum(dim=1, keepdim=True).clamp_min(eps)
    p = p0
    for _ in range(int(steps)):
        propagated = _propagate(p, edge_index, int(num_nodes), 1)
        p = (1.0 - float(alpha)) * p0 + float(alpha) * propagated
        p = p.clamp_min(eps)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    return p


def correct_and_smooth_probabilities(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    correct_alpha: float,
    correct_steps: int,
    smooth_alpha: float,
    smooth_steps: int,
    eps: float = 1e-12,
) -> CorrectSmoothResult:
    base = torch.softmax(logits.to(torch.float32), dim=1)
    labels = labels.to(device=base.device, dtype=torch.long)
    train_idx = train_idx.to(device=base.device, dtype=torch.long)
    residual = torch.zeros_like(base)
    if train_idx.numel() > 0:
        one_hot = torch.nn.functional.one_hot(labels[train_idx], num_classes=int(base.shape[1])).to(torch.float32)
        residual[train_idx] = one_hot - base[train_idx]
    residual = _propagate(residual, edge_index, int(num_nodes), int(correct_steps))
    corrected = base + float(correct_alpha) * residual
    corrected = corrected.clamp_min(eps)
    corrected = corrected / corrected.sum(dim=1, keepdim=True).clamp_min(eps)
    final = smooth_probabilities(probabilities=corrected, edge_index=edge_index, num_nodes=int(num_nodes), alpha=float(smooth_alpha), steps=int(smooth_steps), eps=eps)
    return CorrectSmoothResult(
        logits=torch.log(final.clamp_min(eps)),
        diagnostics={
            "mode": "correct_smooth_lite",
            "normalization": "destination_row",
            "selection_uses_test": False,
            "uses_train_labels": True,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "correct_alpha": float(correct_alpha),
            "correct_steps": int(correct_steps),
            "smooth_alpha": float(smooth_alpha),
            "smooth_steps": int(smooth_steps),
        },
    )
