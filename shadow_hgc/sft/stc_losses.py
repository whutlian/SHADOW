from __future__ import annotations

import torch
import torch.nn.functional as F

from shadow_hgc.sft.stc import BlockSpec


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels.to(torch.long), reduction="none")
    if weights is None:
        return losses.mean()
    weights = weights.to(logits.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def moment_loss(z_syn: torch.Tensor, y_syn: torch.Tensor, z_real: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    loss = z_syn.new_tensor(0.0)
    classes = torch.unique(y_syn.to(torch.long))
    used = 0
    for cls in classes.tolist():
        syn = z_syn[y_syn == int(cls)]
        real = z_real[y_real == int(cls)]
        if syn.numel() == 0 or real.numel() == 0:
            continue
        loss = loss + F.mse_loss(syn.mean(dim=0), real.mean(dim=0))
        if syn.shape[0] > 1 and real.shape[0] > 1:
            loss = loss + F.mse_loss(syn.var(dim=0, unbiased=False), real.var(dim=0, unbiased=False))
        used += 1
    return loss / max(1, used)


def linear_mmd_loss(z_syn: torch.Tensor, y_syn: torch.Tensor, z_real: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    loss = z_syn.new_tensor(0.0)
    classes = torch.unique(y_syn.to(torch.long))
    used = 0
    for cls in classes.tolist():
        syn = z_syn[y_syn == int(cls)]
        real = z_real[y_real == int(cls)]
        if syn.numel() == 0 or real.numel() == 0:
            continue
        loss = loss + F.mse_loss(syn.mean(dim=0), real.mean(dim=0))
        used += 1
    return loss / max(1, used)


def diversity_loss(z_syn: torch.Tensor, y_syn: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    loss = z_syn.new_tensor(0.0)
    used = 0
    for cls in torch.unique(y_syn.to(torch.long)).tolist():
        syn = z_syn[y_syn == int(cls)]
        if syn.shape[0] < 2:
            continue
        distances = torch.pdist(syn, p=2)
        loss = loss + F.relu(float(margin) - distances).pow(2).mean()
        used += 1
    return loss / max(1, used)


def class_prior_loss(y_syn: torch.Tensor, y_real: torch.Tensor, weights: torch.Tensor | None, num_classes: int) -> torch.Tensor:
    device = y_syn.device
    syn_weights = torch.ones_like(y_syn, dtype=torch.float32, device=device) if weights is None else weights.to(device).float()
    syn_counts = torch.zeros(int(num_classes), dtype=torch.float32, device=device)
    syn_counts.scatter_add_(0, y_syn.to(device).long(), syn_weights)
    real_counts = torch.bincount(y_real.to(device).long(), minlength=int(num_classes)).float()
    p_syn = syn_counts / syn_counts.sum().clamp_min(1e-12)
    p_real = real_counts / real_counts.sum().clamp_min(1e-12)
    return F.kl_div((p_syn + 1e-12).log(), p_real, reduction="batchmean")


def block_norm_loss(z_syn: torch.Tensor, z_real: torch.Tensor, blocks: list[BlockSpec] | tuple[BlockSpec, ...]) -> torch.Tensor:
    loss = z_syn.new_tensor(0.0)
    used = 0
    for block in blocks:
        block_slice = block.slice()
        syn = z_syn[:, block_slice]
        real = z_real[:, block_slice]
        loss = loss + F.mse_loss(syn.mean(dim=0), real.mean(dim=0))
        loss = loss + F.mse_loss(syn.std(dim=0, unbiased=False), real.std(dim=0, unbiased=False))
        used += 1
    return loss / max(1, used)


def coverage_loss(logits: torch.Tensor, target_prior: torch.Tensor) -> torch.Tensor:
    pred = F.softmax(logits, dim=1).mean(dim=0)
    target = target_prior.to(logits.device, dtype=pred.dtype)
    target = target / target.sum().clamp_min(1e-12)
    return F.kl_div((pred + 1e-12).log(), target, reduction="batchmean")
