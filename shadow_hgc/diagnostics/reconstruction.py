from __future__ import annotations

import torch


def row_norm_distribution(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "q95": 0.0, "q995": 0.0}
    norms = torch.linalg.norm(x.detach().to(torch.float32), dim=1)
    return {
        "mean": float(norms.mean().item()),
        "median": float(torch.median(norms).item()),
        "q95": float(torch.quantile(norms, 0.95).item()),
        "q995": float(torch.quantile(norms, 0.995).item()),
    }


def reconstruction_error(demand: torch.Tensor, reconstruction: torch.Tensor, *, eps: float = 1e-12) -> float:
    if demand.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(demand - reconstruction).item() / (torch.linalg.norm(demand).item() + eps))
