from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FeatureBlock:
    name: str
    tensor_or_provider: Any
    dim: int
    node_type: str
    role: str


@dataclass
class BlockNormStats:
    name: str
    mean: torch.Tensor
    std: torch.Tensor
    norm_median: float
    norm_p95: float


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    if hasattr(value, "get") and hasattr(value, "shape"):
        rows = torch.arange(int(value.shape[0]), dtype=torch.long)
        return torch.as_tensor(value.get(rows), dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _fit_stats(name: str, x: torch.Tensor, fit_indices: torch.Tensor | None) -> BlockNormStats:
    fit = x if fit_indices is None else x[fit_indices.to(dtype=torch.long, device=x.device)]
    if fit.numel() == 0:
        raise ValueError("fit_indices must select at least one row")
    mean = fit.mean(dim=0)
    std = fit.std(dim=0, unbiased=False).clamp_min(1e-12)
    norms = torch.linalg.vector_norm(fit, dim=1)
    return BlockNormStats(
        name=name,
        mean=mean,
        std=std,
        norm_median=float(torch.quantile(norms, 0.5).item()),
        norm_p95=float(torch.quantile(norms, 0.95).item()),
    )


def _normalize(x: torch.Tensor, stats: BlockNormStats, mode: str) -> torch.Tensor:
    if mode == "none":
        return x.clone()
    out = x
    if mode in {"standardize", "standardize_l2"}:
        out = (out - stats.mean.to(out.device)) / stats.std.to(out.device)
    elif mode not in {"l2"}:
        raise ValueError(f"unsupported block norm mode: {mode}")
    if mode in {"l2", "standardize_l2"}:
        denom = torch.linalg.vector_norm(out, dim=1, keepdim=True).clamp_min(1e-12)
        out = out / denom
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_transform_feature_blocks(
    blocks: list[FeatureBlock],
    *,
    fit_indices: torch.Tensor | None,
    mode: str = "standardize",
) -> tuple[dict[str, torch.Tensor], dict[str, BlockNormStats]]:
    transformed: dict[str, torch.Tensor] = {}
    stats_by_name: dict[str, BlockNormStats] = {}
    for block in blocks:
        x = _as_tensor(block.tensor_or_provider)
        if x.ndim != 2:
            raise ValueError(f"block {block.name} must be a 2D feature matrix")
        if int(x.shape[1]) != int(block.dim):
            raise ValueError(f"block {block.name} declares dim={block.dim}, got {x.shape[1]}")
        stats = _fit_stats(block.name, x, fit_indices)
        transformed[block.name] = _normalize(x, stats, mode)
        stats_by_name[block.name] = stats
    return transformed, stats_by_name
