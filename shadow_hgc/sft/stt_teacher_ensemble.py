from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def average_teacher_probabilities(probs: list[torch.Tensor]) -> torch.Tensor:
    if not probs:
        raise ValueError("at least one teacher probability tensor is required")
    stacked = torch.stack([p.detach().float() for p in probs], dim=0)
    out = stacked.mean(dim=0).clamp_min(0.0)
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


def teacher_ensemble_diagnostics(probs: list[torch.Tensor]) -> dict[str, Any]:
    if not probs:
        return {"teacher_ensemble_size": 0, "teacher_pairwise_kl_mean": "", "teacher_cache_duplicate_detected": False}
    normalized = [p.detach().float().clamp_min(1e-12) / p.detach().float().sum(dim=1, keepdim=True).clamp_min(1e-12) for p in probs]
    kls: list[float] = []
    duplicates = False
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            if torch.allclose(normalized[i], normalized[j], atol=1e-7):
                duplicates = True
            kl = (normalized[i] * (normalized[i].log() - normalized[j].log())).sum(dim=1).mean()
            kls.append(float(kl.item()))
    return {
        "teacher_ensemble_size": len(probs),
        "teacher_pairwise_kl_mean": sum(kls) / len(kls) if kls else 0.0,
        "teacher_pairwise_kl_min": min(kls) if kls else 0.0,
        "teacher_pairwise_kl_max": max(kls) if kls else 0.0,
        "teacher_cache_duplicate_detected": duplicates,
    }


def load_teacher_probs(paths: list[str | Path]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for path in paths:
        arr = np.load(Path(path), mmap_mode="r")
        out.append(torch.from_numpy(np.asarray(arr, dtype=np.float32)))
    return out
