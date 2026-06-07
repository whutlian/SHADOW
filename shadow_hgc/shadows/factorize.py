from __future__ import annotations

import math

import torch


def _clip_rows_by_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    if not math.isfinite(max_norm) or max_norm <= 0:
        return x
    norms = torch.linalg.norm(x, dim=1).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return x * scale.unsqueeze(-1)


def factorize_shadows(
    residual: torch.Tensor,
    *,
    num_shadows: int,
    seed: int = 0,
    sample_weight: torch.Tensor | None = None,
    clip_quantile: float = 0.995,
) -> torch.Tensor:
    """Factorize relation demand rows into signed virtual shadow features."""

    if residual.ndim != 2:
        raise ValueError("residual must have shape [num_rows, feature_dim]")
    if num_shadows <= 0:
        return torch.empty(0, residual.shape[1], dtype=residual.dtype, device=residual.device)
    if residual.shape[0] == 0:
        return torch.zeros(num_shadows, residual.shape[1], dtype=residual.dtype, device=residual.device)

    k = min(num_shadows, residual.shape[0])
    if k == residual.shape[0]:
        centers = residual.detach().clone()
    else:
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                n_init=5,
                batch_size=max(16, min(1024, residual.shape[0])),
            )
            weights_np = sample_weight.detach().cpu().numpy() if sample_weight is not None else None
            km.fit(residual.detach().cpu().numpy(), sample_weight=weights_np)
            centers = torch.as_tensor(km.cluster_centers_, dtype=residual.dtype, device=residual.device)
        except Exception:
            generator = torch.Generator(device=residual.device).manual_seed(seed)
            perm = torch.randperm(residual.shape[0], generator=generator, device=residual.device)[:k]
            centers = residual[perm].detach().clone()

    residual_norms = torch.linalg.norm(residual, dim=1)
    max_norm = float(torch.quantile(residual_norms, clip_quantile).item())
    centers = _clip_rows_by_norm(centers, max_norm)
    if centers.shape[0] < num_shadows:
        pad = torch.zeros(num_shadows - centers.shape[0], residual.shape[1], dtype=residual.dtype, device=residual.device)
        centers = torch.cat([centers, pad], dim=0)
    return centers
