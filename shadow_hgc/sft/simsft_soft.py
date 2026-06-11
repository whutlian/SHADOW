from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SimSFTTable:
    z_syn: torch.Tensor
    y_syn_soft: torch.Tensor
    row_types: list[str]
    diagnostics: dict[str, Any]


def _simplex(probs: torch.Tensor) -> torch.Tensor:
    p = probs.detach().float().clamp_min(0.0)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _class_centroids(features: torch.Tensor, probs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    pred = probs.argmax(dim=1)
    z_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    classes: list[int] = []
    for cls in torch.unique(pred).tolist():
        mask = pred == int(cls)
        if not bool(mask.any()):
            continue
        z_rows.append(features[mask].mean(dim=0))
        y_rows.append(probs[mask].mean(dim=0))
        classes.append(int(cls))
    return z_rows, y_rows, classes


def build_simsft_soft_table(
    *,
    features: torch.Tensor,
    teacher_probs: torch.Tensor,
    num_rows: int,
    method: str = "simsft_soft_centroids",
    seed: int = 42,
    residual_scale: float = 0.5,
) -> SimSFTTable:
    del seed
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    x = features.detach().float().cpu()
    p = _simplex(teacher_probs).cpu()
    z_rows, y_rows, classes = _class_centroids(x, p)
    if not z_rows:
        raise ValueError("teacher_probs must contain at least one row")
    row_types = ["centroid"] * len(z_rows)
    residual_count = 0
    global_mean = x.mean(dim=0)
    global_norm = float(global_mean.norm().item())
    residual_clip = float(torch.quantile((x - global_mean).norm(dim=1), 0.90).item()) if x.shape[0] > 1 else 0.0
    if "residual" in method or "lowrank" in method or "boundary" in method:
        pred = p.argmax(dim=1)
        for cls, centroid, soft in zip(classes, list(z_rows), list(y_rows)):
            if len(z_rows) >= num_rows:
                break
            mask_idx = torch.nonzero(pred == cls, as_tuple=False).flatten()
            if mask_idx.numel() == 0:
                continue
            residual = x[mask_idx] - centroid
            norms = residual.norm(dim=1)
            src = mask_idx[int(torch.argmax(norms).item())]
            direction = x[src] - centroid
            norm = direction.norm().clamp_min(1e-12)
            step = min(float(norm.item()) * float(residual_scale), residual_clip)
            z_rows.append(centroid + direction / norm * step)
            y_rows.append(soft)
            row_types.append("residual")
            residual_count += 1
    if "boundary" in method:
        margin = torch.topk(p, k=min(2, p.shape[1]), dim=1).values
        boundary_score = margin[:, 0] - (margin[:, 1] if margin.shape[1] > 1 else 0.0)
        for idx in torch.argsort(boundary_score, descending=False).tolist():
            if len(z_rows) >= num_rows:
                break
            z_rows.append(x[int(idx)])
            y_rows.append(p[int(idx)])
            row_types.append("boundary")
    while len(z_rows) < num_rows:
        src = len(z_rows) % x.shape[0]
        z_rows.append(x[src])
        y_rows.append(p[src])
        row_types.append("fallback_real")
    z_syn = torch.stack(z_rows[:num_rows], dim=0)
    y_syn_soft = _simplex(torch.stack(y_rows[:num_rows], dim=0))
    diagnostics = {
        "uses_full_covariance": False,
        "uses_exact_pairwise": False,
        "residual_row_count": residual_count,
        "residual_norm_clip": residual_clip,
        "global_mean_norm": global_norm,
        "row_type_counts": {name: row_types[:num_rows].count(name) for name in sorted(set(row_types[:num_rows]))},
    }
    return SimSFTTable(z_syn=z_syn, y_syn_soft=y_syn_soft, row_types=row_types[:num_rows], diagnostics=diagnostics)


def simsft_promotion_status(*, ratio: float, accuracy: float) -> tuple[str, str]:
    if abs(float(ratio) - 0.001) < 1e-12 and float(accuracy) >= 0.923:
        return "promoted", "simsft_table_only_gate_met"
    if abs(float(ratio) - 0.005) < 1e-12 and float(accuracy) >= 0.928:
        return "promoted", "simsft_table_only_gate_met"
    return "not_promoted", "simsft_table_only_gate_not_met"
