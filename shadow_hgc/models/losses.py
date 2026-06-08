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
    class_prior: torch.Tensor | None = None,
    logit_adjustment_tau: float = 1.0,
) -> torch.Tensor:
    """Prototype CE variants, defaulting to the cell-weighted empirical risk."""

    if loss_type == "sqrt_weighted_logit_adjusted":
        if class_prior is None:
            counts = torch.bincount(labels.to(torch.long), minlength=logits.shape[1]).to(logits.device, logits.dtype)
            class_prior = counts / counts.sum().clamp_min(1e-12)
        class_prior = class_prior.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-12)
        logits = logits - float(logit_adjustment_tau) * torch.log(class_prior).unsqueeze(0)

    ce = F.cross_entropy(logits, labels, reduction="none")
    if weights is None:
        weights = torch.ones_like(ce)
    weights = weights.to(dtype=ce.dtype, device=ce.device)

    if loss_type == "weighted":
        effective = weights
    elif loss_type == "sqrt_weighted":
        effective = torch.sqrt(weights.clamp_min(0.0))
    elif loss_type == "sqrt_weighted_logit_adjusted":
        effective = torch.sqrt(weights.clamp_min(0.0))
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
