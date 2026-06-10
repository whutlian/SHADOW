from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BalancedTrainerConfig:
    recipe: str = "balanced_adamw_label_smoothing_mixup"
    balanced_batches: bool = True
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.0
    ema: bool = False
    swa: bool = False
    logit_adjustment: bool = False
    lr: float = 0.003
    weight_decay: float = 1e-4

    def diagnostics(self) -> dict[str, Any]:
        return {
            "trainer_recipe": self.recipe,
            "trainer_balanced_batches": bool(self.balanced_batches),
            "trainer_label_smoothing": float(self.label_smoothing),
            "trainer_mixup_alpha": float(self.mixup_alpha),
            "trainer_ema": bool(self.ema),
            "trainer_swa": bool(self.swa),
            "trainer_logit_adjustment": bool(self.logit_adjustment),
        }


def balanced_batch_order(rows: torch.Tensor, labels: torch.Tensor, *, seed: int = 42) -> torch.Tensor:
    rows = rows.to(torch.long).cpu()
    labels = labels.to(torch.long).cpu()
    if rows.numel() == 0:
        return rows
    generator = torch.Generator().manual_seed(int(seed))
    per_class: list[torch.Tensor] = []
    for cls in torch.unique(labels[rows], sorted=True):
        cls_rows = rows[labels[rows] == cls]
        per_class.append(cls_rows[torch.randperm(cls_rows.numel(), generator=generator)])
    output: list[torch.Tensor] = []
    max_len = max(int(chunk.numel()) for chunk in per_class)
    for idx in range(max_len):
        for chunk in per_class:
            if idx < int(chunk.numel()):
                output.append(chunk[idx : idx + 1])
    return torch.cat(output).to(torch.long)


def condensed_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_labels: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_adjustment: bool = False,
    class_balanced: bool = False,
) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long)
    train_labels = train_labels.to(device=logits.device, dtype=torch.long)
    adjusted = logits
    if bool(logit_adjustment):
        counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
        prior = counts / counts.sum().clamp_min(1e-12)
        adjusted = logits - torch.log(prior.clamp_min(1e-12)).unsqueeze(0)
    ce = F.cross_entropy(adjusted, labels, reduction="none", label_smoothing=float(label_smoothing))
    if not bool(class_balanced):
        return ce.mean()
    counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
    weights = (1.0 / counts)[labels]
    return (weights * ce).sum() / weights.sum().clamp_min(1e-12)


def within_class_sft_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    del seed
    if float(alpha) <= 0.0:
        return features, labels
    mixed = features.clone()
    labels = labels.to(torch.long)
    lam = 0.5
    for cls in torch.unique(labels, sorted=True):
        idx = torch.nonzero(labels == cls, as_tuple=False).view(-1)
        if idx.numel() <= 1:
            continue
        partner = torch.roll(idx, shifts=1)
        mixed[idx] = lam * features[idx] + (1.0 - lam) * features[partner]
    return mixed, labels
