from __future__ import annotations

import torch

from shadow_hgc.features.pseudo_scap import PseudoLabelResult


def build_t1_pseudo_labels(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    threshold: float,
    pseudo_weight: float,
    temperature: float = 1.0,
) -> PseudoLabelResult:
    z = logits.to(torch.float32)
    probs = torch.softmax(z / max(float(temperature), 1e-12), dim=1)
    confidence = probs.max(dim=1).values
    labels = labels.to(device=z.device, dtype=torch.long)
    train_idx = train_idx.to(device=z.device, dtype=torch.long)
    train_mask = torch.zeros(int(z.shape[0]), dtype=torch.bool, device=z.device)
    if train_idx.numel() > 0:
        train_mask[train_idx] = True
    weights = torch.zeros(int(z.shape[0]), dtype=torch.float32, device=z.device)
    pseudo = torch.zeros_like(probs)
    keep = (~train_mask) & (confidence >= float(threshold))
    pseudo[keep] = probs[keep]
    weights[keep] = float(pseudo_weight)
    if train_idx.numel() > 0:
        pseudo[train_idx] = torch.nn.functional.one_hot(labels[train_idx], num_classes=int(z.shape[1])).to(torch.float32)
        weights[train_idx] = 1.0
    nontrain_total = int((~train_mask).sum().item())
    nontrain_used = int(keep.sum().item())
    return PseudoLabelResult(
        pseudo=pseudo,
        weights=weights,
        confidence=confidence,
        diagnostics={
            "threshold": float(threshold),
            "pseudo_weight": float(pseudo_weight),
            "temperature": float(temperature),
            "train_override_count": int(train_idx.numel()),
            "nontrain_used_count": nontrain_used,
            "pseudo_coverage": 0.0 if nontrain_total == 0 else float(nontrain_used / nontrain_total),
            "mean_confidence": float(confidence.mean().item()) if confidence.numel() else 0.0,
            "median_confidence": float(confidence.median().item()) if confidence.numel() else 0.0,
            "uses_train_labels": True,
            "uses_validation_labels": False,
            "uses_test_labels": False,
        },
    )
