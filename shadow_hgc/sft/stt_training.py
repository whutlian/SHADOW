from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _as_probs(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().float()
    row_sum = x.sum(dim=1, keepdim=True)
    if bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)):
        return x / row_sum.clamp_min(1e-12)
    return torch.softmax(x, dim=1)


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor, *, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    logp = F.log_softmax(logits.float() / temp, dim=1)
    y = _as_probs(targets).to(logits.device)
    return -(y * logp).sum(dim=1).mean() * (temp * temp)


def stt_soft_target_loss(
    logits: torch.Tensor,
    y_soft: torch.Tensor,
    *,
    y_hard: torch.Tensor | None = None,
    hard_anchor_mask: torch.Tensor | None = None,
    teacher_prior: torch.Tensor | None = None,
    temperature: float = 2.0,
    lambda_soft: float = 1.0,
    lambda_hard: float = 0.0,
    lambda_prior: float = 0.0,
    lambda_calib: float = 0.0,
    lambda_mix: float = 0.0,
    mixup_alpha: float = 0.4,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """STT condensed-table loss.

    Teacher probabilities are soft targets only. They are never interpreted as
    model input features here.
    """

    if logits.ndim != 2 or y_soft.ndim != 2:
        raise ValueError("logits and y_soft must be rank-2 tensors")
    if logits.shape != y_soft.shape:
        raise ValueError("logits and y_soft must have identical shape")

    y = _as_probs(y_soft).to(logits.device)
    total = logits.new_tensor(0.0)

    soft_loss = _soft_ce(logits, y, temperature=float(temperature))
    total = total + float(lambda_soft) * soft_loss

    hard_loss = logits.new_tensor(0.0)
    if y_hard is not None and float(lambda_hard) != 0.0:
        mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device) if hard_anchor_mask is None else hard_anchor_mask.to(logits.device).bool()
        hard = y_hard.to(logits.device).long()
        mask = mask & (hard >= 0)
        if bool(mask.any().item()):
            hard_loss = F.cross_entropy(logits[mask], hard[mask])
            total = total + float(lambda_hard) * hard_loss

    prior_loss = logits.new_tensor(0.0)
    pred_probs = torch.softmax(logits.float(), dim=1)
    if teacher_prior is not None and float(lambda_prior) != 0.0:
        target_prior = _as_probs(teacher_prior.detach().float().view(1, -1)).view(-1).to(logits.device)
        pred_prior = pred_probs.mean(dim=0).clamp_min(1e-12)
        prior_loss = F.kl_div(pred_prior.log(), target_prior, reduction="sum")
        total = total + float(lambda_prior) * prior_loss

    calib_loss = logits.new_tensor(0.0)
    if float(lambda_calib) != 0.0:
        pred_conf = pred_probs.max(dim=1).values
        teacher_conf = y.max(dim=1).values
        calib_loss = F.mse_loss(pred_conf, teacher_conf)
        total = total + float(lambda_calib) * calib_loss

    mix_loss = logits.new_tensor(0.0)
    virtual_mixup_count = 0
    if float(lambda_mix) != 0.0 and logits.shape[0] > 1:
        gen = torch.Generator(device=logits.device if logits.device.type == "cuda" else "cpu")
        gen.manual_seed(int(seed))
        perm = torch.randperm(logits.shape[0], generator=gen, device=logits.device)
        alpha = max(float(mixup_alpha), 1e-6)
        beta = torch.distributions.Beta(alpha, alpha)
        lam = beta.sample((logits.shape[0],)).to(logits.device).view(-1, 1)
        mixed_logits = lam * logits + (1.0 - lam) * logits[perm]
        mixed_y = lam * y + (1.0 - lam) * y[perm]
        mix_loss = _soft_ce(mixed_logits, mixed_y, temperature=float(temperature))
        total = total + float(lambda_mix) * mix_loss
        virtual_mixup_count = int(logits.shape[0])

    parts = {
        "soft_loss": float(soft_loss.detach().cpu().item()),
        "hard_loss": float(hard_loss.detach().cpu().item()),
        "prior_loss": float(prior_loss.detach().cpu().item()),
        "calib_loss": float(calib_loss.detach().cpu().item()),
        "mix_loss": float(mix_loss.detach().cpu().item()),
        "virtual_mixup_count": virtual_mixup_count,
    }
    return total, parts
