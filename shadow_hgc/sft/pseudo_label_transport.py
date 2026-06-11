from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PLTCSelection:
    selected_idx: torch.Tensor
    diagnostics: dict[str, Any]


def build_pltc_soft_labels(teacher_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = teacher_probs.to(torch.float32).clamp_min(float(eps))
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(float(eps))


def confidence_bins(teacher_probs: torch.Tensor) -> np.ndarray:
    conf = teacher_probs.to(torch.float32).max(dim=1).values.cpu().numpy()
    labels = np.empty(conf.shape[0], dtype=object)
    labels[conf >= 0.80] = "high"
    labels[(conf >= 0.50) & (conf < 0.80)] = "medium"
    labels[conf < 0.50] = "low"
    return labels


def select_pltc_indices(teacher_probs: torch.Tensor, *, total_budget: int, seed: int = 42) -> PLTCSelection:
    probs = build_pltc_soft_labels(teacher_probs)
    n = int(probs.shape[0])
    budget = min(max(1, int(total_budget)), n)
    conf = probs.max(dim=1).values
    pred = probs.argmax(dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    score = 0.50 * conf + 0.30 * entropy / entropy.max().clamp_min(1e-12) + 0.20 * (pred.to(torch.float32) / max(1, probs.shape[1] - 1))
    generator = torch.Generator().manual_seed(int(seed))
    jitter = torch.rand(n, generator=generator) * 1e-6
    selected = torch.argsort(score + jitter, descending=True)[:budget].to(torch.long)
    bins = confidence_bins(probs)
    unique_classes = torch.unique(pred[selected]).numel()
    return PLTCSelection(
        selected_idx=selected,
        diagnostics={
            "pltc_num_soft_nodes": int(selected.numel()),
            "pltc_confidence_min": float(conf[selected].min().item()) if selected.numel() else "",
            "pltc_confidence_max": float(conf[selected].max().item()) if selected.numel() else "",
            "pltc_confidence_bins": json.dumps({name: int((bins[selected.numpy()] == name).sum()) for name in ["high", "medium", "low"]}, sort_keys=True),
            "pltc_soft_class_coverage": int(unique_classes),
            "uses_teacher_logits": True,
            "uses_logits_as_input": False,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "promotion_track": "sota_chase",
        },
    )
