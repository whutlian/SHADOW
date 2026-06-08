from __future__ import annotations

import torch


def class_collapse_diagnostics(
    pred: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    *,
    num_classes: int,
    top_k: int = 5,
) -> dict:
    if idx.numel() == 0:
        return {
            "predicted_class_count": 0,
            "prediction_entropy": 0.0,
            "top_predicted_classes": [],
            "per_class_accuracy": {},
            "per_class_support": {},
        }
    selected_pred = pred[idx].to(torch.long)
    selected_labels = labels[idx].to(torch.long)
    hist = torch.bincount(selected_pred.clamp_min(0), minlength=num_classes).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * torch.log(probs[probs > 0])).sum().item())
    top_values, top_indices = torch.topk(hist, k=min(top_k, int(hist.numel())))
    per_class_accuracy = {}
    per_class_support = {}
    for class_id in range(num_classes):
        mask = selected_labels == class_id
        support = int(mask.sum().item())
        per_class_support[str(class_id)] = support
        if support > 0:
            per_class_accuracy[str(class_id)] = float((selected_pred[mask] == class_id).to(torch.float32).mean().item())
        else:
            per_class_accuracy[str(class_id)] = None
    return {
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
        "top_predicted_classes": [
            {"class_id": int(class_id.item()), "count": int(count.item())}
            for class_id, count in zip(top_indices, top_values)
            if int(count.item()) > 0
        ],
        "per_class_accuracy": per_class_accuracy,
        "per_class_support": per_class_support,
    }
