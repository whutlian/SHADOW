from __future__ import annotations

import math

import torch


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.to(torch.float32), q).item())


def _svdvals(matrix: torch.Tensor, rank_k: int | None = None) -> torch.Tensor:
    matrix = matrix.detach().to(torch.float32)
    if matrix.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=matrix.device)
    min_dim = min(matrix.shape)
    if min_dim == 0:
        return torch.empty(0, dtype=torch.float32, device=matrix.device)
    if rank_k is None or min_dim <= rank_k or matrix.numel() <= 2_000_000:
        return torch.linalg.svdvals(matrix)
    q = max(1, min(int(rank_k), min_dim))
    _, singular_values, _ = torch.pca_lowrank(matrix, q=q, center=False)
    return singular_values


def relation_rank_diagnostics(matrix: torch.Tensor, *, rank_k: int | None = None, eps: float = 1e-12) -> dict[str, float]:
    """Compute train-target relation demand rank and norm diagnostics."""

    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [num_rows, feature_dim]")
    matrix = matrix.detach().to(torch.float32)
    fro_sq = float(torch.sum(matrix * matrix).item())
    singular_values = _svdvals(matrix, rank_k=rank_k)
    top_sq = float((singular_values[0] ** 2).item()) if singular_values.numel() else 0.0
    stable_rank = fro_sq / (top_sq + eps) if fro_sq > 0.0 else 0.0
    if singular_values.numel() and fro_sq > 0.0:
        energy = singular_values.square()
        probs = energy / energy.sum().clamp_min(eps)
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
        entropy_effective_rank = float(torch.exp(entropy).item())
    else:
        entropy_effective_rank = 0.0
    norms = torch.linalg.norm(matrix, dim=1) if matrix.shape[0] else torch.empty(0, device=matrix.device)
    return {
        "stable_rank": float(stable_rank) if math.isfinite(stable_rank) else 0.0,
        "entropy_effective_rank": float(entropy_effective_rank) if math.isfinite(entropy_effective_rank) else 0.0,
        "relation_demand_norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
        "relation_demand_norm_median": float(torch.median(norms).item()) if norms.numel() else 0.0,
        "relation_demand_norm_q95": _quantile(norms, 0.95),
        "relation_demand_norm_q995": _quantile(norms, 0.995),
    }
