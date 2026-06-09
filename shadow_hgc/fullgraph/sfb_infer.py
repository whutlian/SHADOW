from __future__ import annotations

import torch

from shadow_hgc.fullgraph.sfb_model import BlockGatedResidualTableModel


@torch.no_grad()
def predict_logits(
    model: BlockGatedResidualTableModel,
    blocks: dict[str, torch.Tensor],
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    model.eval()
    if batch_size is None:
        return model(blocks)
    num_rows = int(next(iter(blocks.values())).shape[0])
    chunks: list[torch.Tensor] = []
    for start in range(0, num_rows, int(batch_size)):
        end = min(num_rows, start + int(batch_size))
        chunks.append(model({name: value[start:end] for name, value in blocks.items()}))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor) -> dict[str, float | int]:
    selected = logits[rows]
    y = labels[rows].long()
    pred = selected.argmax(dim=1)
    return {
        "accuracy": float((pred == y).to(torch.float32).mean().item()) if y.numel() else 0.0,
        "predicted_class_count": int(torch.unique(pred).numel()),
    }
