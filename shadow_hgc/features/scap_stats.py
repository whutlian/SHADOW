from __future__ import annotations

import torch


def scap_row_stats(dense: torch.Tensor) -> dict[str, torch.Tensor | dict]:
    support = (dense.abs() > 0).sum(dim=1).to(torch.float32)
    mass = dense.abs()
    probs = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    max_affinity = dense.abs().max(dim=1).values if dense.numel() else torch.zeros(dense.shape[0])
    missing = support == 0
    return {
        "support_count": support,
        "row_entropy": entropy,
        "max_affinity": max_affinity,
        "missingness": missing,
        "support_count_stats": {
            "mean": float(support.mean().item()) if support.numel() else 0.0,
            "max": float(support.max().item()) if support.numel() else 0.0,
        },
        "entropy_stats": {
            "mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
            "max": float(entropy.max().item()) if entropy.numel() else 0.0,
        },
    }
