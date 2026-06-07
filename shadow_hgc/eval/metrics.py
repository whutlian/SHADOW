from __future__ import annotations

import torch


def macro_f1_score(pred: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None) -> float:
    pred = pred.to(torch.long).flatten()
    label = label.to(torch.long).flatten()
    if num_classes is None:
        num_classes = int(torch.cat([pred, label]).max().item()) + 1 if pred.numel() else 0
    scores = []
    for class_id in range(num_classes):
        tp = ((pred == class_id) & (label == class_id)).sum().to(torch.float32)
        fp = ((pred == class_id) & (label != class_id)).sum().to(torch.float32)
        fn = ((pred != class_id) & (label == class_id)).sum().to(torch.float32)
        denom = 2 * tp + fp + fn
        if denom.item() == 0:
            continue
        scores.append((2 * tp / denom).item())
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
