from __future__ import annotations

import torch


def shadow_reconstruction_error(demand: torch.Tensor, shadow_features: torch.Tensor, assignment: torch.Tensor) -> float:
    if demand.numel() == 0:
        return 0.0
    recon = shadow_features[assignment]
    return float(torch.linalg.norm(demand - recon).item() / (torch.linalg.norm(demand).item() + 1e-12))


def feature_norm_summary(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    norms = torch.linalg.norm(x, dim=1)
    return {
        "min": float(norms.min().item()),
        "median": float(torch.median(norms).item()),
        "max": float(norms.max().item()),
    }
