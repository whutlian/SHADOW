from __future__ import annotations

import torch


def aggregate_blocks_by_prototype(blocks: dict[str, torch.Tensor], assignments: torch.Tensor) -> dict[str, torch.Tensor]:
    assignments = assignments.to(torch.long)
    num_proto = int(assignments.max().item()) + 1 if assignments.numel() else 0
    counts = torch.bincount(assignments, minlength=num_proto).to(torch.float32).clamp_min(1.0)
    out: dict[str, torch.Tensor] = {}
    for name, block in blocks.items():
        proto = torch.zeros(num_proto, int(block.shape[1]), dtype=torch.float32)
        proto.index_add_(0, assignments, block.to(torch.float32))
        out[name] = proto / counts.unsqueeze(1)
    return out
