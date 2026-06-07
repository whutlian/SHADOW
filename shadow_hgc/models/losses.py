from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    loss_type: str = "weighted",
    clip_value: float | None = None,
) -> torch.Tensor:
    """Prototype CE variants, defaulting to the cell-weighted empirical risk."""

    ce = F.cross_entropy(logits, labels, reduction="none")
    if weights is None:
        weights = torch.ones_like(ce)
    weights = weights.to(dtype=ce.dtype, device=ce.device)

    if loss_type == "weighted":
        effective = weights
    elif loss_type == "unweighted":
        effective = torch.ones_like(weights)
    elif loss_type == "clipped":
        if clip_value is None:
            clip_value = float(torch.quantile(weights.detach(), 0.95).item())
        effective = torch.clamp(weights, max=clip_value)
    elif loss_type == "class_balanced":
        effective = torch.zeros_like(weights)
        for label in labels.unique():
            mask = labels == label
            effective[mask] = weights[mask] / weights[mask].sum().clamp_min(1e-12)
    else:
        raise ValueError(f"unknown prototype loss type: {loss_type}")

    return (effective * ce).sum() / effective.sum().clamp_min(1e-12)
