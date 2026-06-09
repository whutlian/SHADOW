from __future__ import annotations

import torch


def nearest_shadow_block_reconstruction(block: torch.Tensor, shadows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if shadows.ndim != 2 or block.ndim != 2:
        raise ValueError("block and shadows must both be 2D")
    if int(block.shape[1]) != int(shadows.shape[1]):
        raise ValueError("block and shadows must have the same feature dimension")
    distances = torch.cdist(block.to(torch.float32), shadows.to(torch.float32))
    assignment = distances.argmin(dim=1).to(torch.long)
    return shadows[assignment].to(torch.float32), assignment
