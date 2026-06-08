from __future__ import annotations

import torch


def fixed_block_projection(x: torch.Tensor, *, out_dim: int, seed: int) -> torch.Tensor:
    if x.shape[1] <= out_dim:
        return x
    generator = torch.Generator(device=x.device).manual_seed(seed)
    weight = torch.randn(x.shape[1], out_dim, generator=generator, device=x.device, dtype=x.dtype)
    weight = weight / max(1.0, float(x.shape[1]) ** 0.5)
    return x @ weight
