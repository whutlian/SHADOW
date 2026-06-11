from __future__ import annotations

import json
from typing import Any

import torch


def qoc_pltc_split(num_codewords: int) -> dict[str, int]:
    total = int(num_codewords)
    confident = int(round(total * 0.50))
    uncertain = int(round(total * 0.25))
    structural = int(round(total * 0.15))
    rare = max(0, total - confident - uncertain - structural)
    return {
        "confident_pseudo_class_codewords": confident,
        "uncertain_boundary_codewords": uncertain,
        "structural_high_degree_codewords": structural,
        "rare_class_bucket_codewords": rare,
    }


def aggregate_teacher_soft_labels(assignments: torch.Tensor, teacher_probs: torch.Tensor, *, num_codewords: int) -> tuple[torch.Tensor, dict[str, Any]]:
    assign = assignments.to(torch.long).cpu()
    probs = teacher_probs.to(torch.float32).cpu()
    if probs.ndim != 2:
        raise ValueError("teacher_probs must have shape [N, C]")
    sums = torch.zeros(int(num_codewords), probs.shape[1], dtype=torch.float32)
    counts = torch.bincount(assign.clamp_min(0), minlength=int(num_codewords)).to(torch.float32)[: int(num_codewords)]
    valid = (assign >= 0) & (assign < int(num_codewords))
    sums.index_add_(0, assign[valid], probs[valid])
    nonzero = counts > 0
    sums[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)
    sums[~nonzero] = 1.0 / float(probs.shape[1])
    sums = sums / sums.sum(dim=1, keepdim=True).clamp_min(1e-12)
    confidence = sums.max(dim=1).values
    bins = {
        "high": int((confidence >= 0.8).sum().item()),
        "medium": int(((confidence >= 0.5) & (confidence < 0.8)).sum().item()),
        "low": int((confidence < 0.5).sum().item()),
    }
    entropy = float((-(sums.clamp_min(1e-12) * sums.clamp_min(1e-12).log()).sum(dim=1)).mean().item())
    return sums, {
        "uses_teacher_logits": True,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "soft_label_entropy_mean": entropy,
        "confidence_bin_counts": json.dumps(bins, sort_keys=True),
    }
