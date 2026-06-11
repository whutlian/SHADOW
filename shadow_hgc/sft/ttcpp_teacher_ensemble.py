from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemperatureCalibrationResult:
    temperature: float
    valid_nll: float
    uses_valid_labels_as_input: bool = False


@dataclass(frozen=True)
class EnsembleProbabilities:
    probs: torch.Tensor
    disagreement: torch.Tensor
    member_probs: list[torch.Tensor]


def _as_probs(logits_or_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    x = logits_or_probs.detach().float()
    row_sum = x.sum(dim=1, keepdim=True)
    if bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)):
        return x / row_sum.clamp_min(1e-12)
    return torch.softmax(x / float(temperature), dim=1)


def calibrate_teacher_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_idx: torch.Tensor,
    temperatures: list[float],
) -> TemperatureCalibrationResult:
    if not temperatures:
        raise ValueError("temperature grid cannot be empty")
    y = labels[valid_idx].to(torch.long)
    best_temp = float(temperatures[0])
    best_nll = float("inf")
    for temp in temperatures:
        logp = F.log_softmax(logits[valid_idx].float() / float(temp), dim=1)
        nll = float(F.nll_loss(logp, y, reduction="mean").item())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return TemperatureCalibrationResult(temperature=best_temp, valid_nll=best_nll)


def build_ensemble_probabilities(logits_or_probs: list[torch.Tensor], *, temperatures: list[float] | None = None) -> EnsembleProbabilities:
    if not logits_or_probs:
        raise ValueError("at least one teacher is required")
    if temperatures is None:
        temperatures = [1.0] * len(logits_or_probs)
    if len(temperatures) != len(logits_or_probs):
        raise ValueError("temperatures length must match teacher count")
    member_probs = [_as_probs(logits, temp) for logits, temp in zip(logits_or_probs, temperatures)]
    probs = torch.stack(member_probs, dim=0).mean(dim=0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    disagreement_terms = []
    for member in member_probs:
        disagreement_terms.append((member * (member.clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(dim=1))
    disagreement = torch.stack(disagreement_terms, dim=0).mean(dim=0)
    return EnsembleProbabilities(probs=probs, disagreement=disagreement, member_probs=member_probs)


def compute_teacher_diagnostics(probs: torch.Tensor, disagreement: torch.Tensor | None = None) -> dict[str, Any]:
    p = _as_probs(probs)
    entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(p, k=min(2, p.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    pred = p.argmax(dim=1)
    return {
        "teacher_entropy_mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
        "teacher_margin_mean": float(margin.mean().item()) if margin.numel() else 0.0,
        "teacher_disagreement_mean": round(float(disagreement.float().mean().item()), 6) if disagreement is not None and disagreement.numel() else 0.0,
        "predicted_classes": int(pred.unique().numel()) if pred.numel() else 0,
        "teacher_allnode_prior": [float(v) for v in p.mean(dim=0).tolist()],
    }


def tensor_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_teacher_cache_manifest(path: str | Path, *, rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"teachers": rows, "diagnostics": diagnostics}, indent=2, sort_keys=True), encoding="utf-8")
    return target
