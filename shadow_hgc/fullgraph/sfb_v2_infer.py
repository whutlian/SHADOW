from __future__ import annotations

import torch

from shadow_hgc.models.block_gated_table import BlockGatedTableModel


@torch.no_grad()
def predict_sfb_v2_logits(
    model: BlockGatedTableModel,
    blocks: dict[str, torch.Tensor],
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    model.eval()
    if batch_size is None:
        return model(blocks)
    n = int(next(iter(blocks.values())).shape[0])
    chunks: list[torch.Tensor] = []
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        chunks.append(model({name: value[start:end] for name, value in blocks.items()}))
    return torch.cat(chunks, dim=0)
