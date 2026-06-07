from __future__ import annotations

import torch


def calibrate_shadow_norm(
    demand: torch.Tensor,
    shadow_features: torch.Tensor,
    assignment: torch.Tensor,
    *,
    enabled: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float]:
    if not enabled or demand.numel() == 0:
        return shadow_features, 1.0
    recon = shadow_features[assignment]
    ratio = torch.linalg.norm(demand, dim=1) / torch.linalg.norm(recon, dim=1).clamp_min(eps)
    gamma = float(torch.median(ratio).clamp(0.5, 2.0).item())
    return gamma * shadow_features, gamma
