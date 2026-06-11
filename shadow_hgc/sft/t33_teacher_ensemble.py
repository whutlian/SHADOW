from __future__ import annotations

import hashlib
from typing import Any

import torch


def _as_probs(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().float()
    row_sum = x.sum(dim=1, keepdim=True)
    if bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)):
        return x / row_sum.clamp_min(1e-12)
    return torch.softmax(x, dim=1)


def cache_tensor_hash(values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(values.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pp = p.clamp_min(1e-12)
    qq = q.clamp_min(1e-12)
    return (pp * (pp.log() - qq.log())).sum(dim=1)


def teacher_ensemble_diversity(probs_list: list[torch.Tensor], *, teacher_ids: list[str] | None = None) -> dict[str, Any]:
    if not probs_list:
        return {
            "teacher_ensemble_size": 0,
            "teacher_cache_duplicate_detected": False,
            "teacher_disagreement_mean": 0.0,
            "teacher_pairwise_kl_mean": 0.0,
            "teacher_pairwise_kl_min": 0.0,
            "teacher_pairwise_kl_max": 0.0,
            "teacher_cache_hashes": "",
            "ensemble_failure_reason": "missing_teacher_cache",
        }
    probs = [_as_probs(p).cpu() for p in probs_list]
    hashes = [cache_tensor_hash(p) for p in probs]
    mean_probs = torch.stack(probs, dim=0).mean(dim=0)
    disagreement = torch.stack([_kl(p, mean_probs) for p in probs], dim=0).mean(dim=0)
    pairwise: list[float] = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            pairwise.append(float((0.5 * (_kl(probs[i], probs[j]) + _kl(probs[j], probs[i]))).mean().item()))
    duplicate = len(set(hashes)) != len(hashes)
    pairwise_mean = float(sum(pairwise) / len(pairwise)) if pairwise else 0.0
    failure = ""
    disagreement_mean = float(disagreement.mean().item()) if disagreement.numel() else 0.0
    if len(probs) > 1 and (duplicate or disagreement_mean <= 1e-6 or pairwise_mean <= 1e-12):
        failure = "ensemble_not_diverse"
    return {
        "teacher_ensemble_size": len(probs),
        "teacher_ids": ";".join(teacher_ids or [str(i) for i in range(len(probs))]),
        "teacher_cache_duplicate_detected": duplicate,
        "teacher_cache_hashes": ";".join(hashes),
        "teacher_disagreement_mean": disagreement_mean,
        "teacher_pairwise_kl_mean": pairwise_mean,
        "teacher_pairwise_kl_min": min(pairwise) if pairwise else 0.0,
        "teacher_pairwise_kl_max": max(pairwise) if pairwise else 0.0,
        "ensemble_failure_reason": failure,
    }
