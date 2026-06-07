from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StandardizerStats:
    mean: torch.Tensor
    std: torch.Tensor


def fixed_random_projection(x: torch.Tensor, *, out_dim: int, seed: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape [num_nodes, feature_dim]")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    projection = torch.randn(x.shape[1], out_dim, generator=generator, device=x.device, dtype=x.dtype)
    projection = projection / (x.shape[1] ** 0.5)
    return x @ projection


def fit_standardizer(x: torch.Tensor, *, rows: torch.Tensor | None = None, eps: float = 1e-6) -> StandardizerStats:
    scoped = x if rows is None else x[rows]
    return StandardizerStats(mean=scoped.mean(dim=0), std=scoped.std(dim=0, unbiased=False).clamp_min(eps))


def standardize(x: torch.Tensor, stats: StandardizerStats) -> torch.Tensor:
    return (x - stats.mean.to(x.device)) / stats.std.to(x.device)
