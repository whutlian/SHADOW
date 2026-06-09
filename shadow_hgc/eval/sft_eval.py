from __future__ import annotations

import torch

from shadow_hgc.eval.metrics import macro_f1_score


def sft_metrics(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, *, num_classes: int) -> dict:
    pred = logits.argmax(dim=1).to(torch.long)
    selected = pred[rows]
    y = labels[rows].to(torch.long)
    hist = torch.bincount(selected.clamp_min(0), minlength=int(num_classes)).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if hist.numel() else 0.0
    return {
        "accuracy": float((selected == y).to(torch.float32).mean().item()) if y.numel() else 0.0,
        "macro_f1": macro_f1_score(selected, y, num_classes=num_classes),
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
    }


def predict_sft_logits(model, blocks: dict[str, torch.Tensor], *, batch_size: int | None = None) -> torch.Tensor:
    model.eval()
    if batch_size is None:
        with torch.no_grad():
            return model(blocks).detach().cpu()
    num_rows = next(iter(blocks.values())).shape[0]
    chunks = []
    with torch.no_grad():
        for start in range(0, int(num_rows), int(batch_size)):
            part = {name: value[start : start + int(batch_size)] for name, value in blocks.items()}
            chunks.append(model(part).detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, model.num_classes)
