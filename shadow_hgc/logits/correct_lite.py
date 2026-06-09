from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class LogitCorrectResult:
    logits: torch.Tensor
    diagnostics: dict


def _ensure_logits(logits: torch.Tensor, num_nodes: int) -> torch.Tensor:
    z = logits.to(torch.float32)
    if z.ndim != 2:
        raise ValueError("logits must have shape [num_nodes, num_classes]")
    if int(z.shape[0]) != int(num_nodes):
        raise ValueError("num_nodes must match logits rows")
    return z


def _propagate_once(
    *,
    edge_index: torch.Tensor,
    values: torch.Tensor,
    num_nodes: int,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    out = torch.zeros(int(num_nodes), int(values.shape[1]), dtype=torch.float32, device=values.device)
    if edge_index.numel() == 0:
        return values.clone()
    edge_index = edge_index.to(device=values.device, dtype=torch.long)
    alpha = destination_row_normalize(edge_index, int(num_nodes), raw_edge_weight=raw_edge_weight).to(device=values.device, dtype=torch.float32)
    for start in range(0, int(edge_index.shape[1]), int(chunk_size)):
        end = min(int(edge_index.shape[1]), start + int(chunk_size))
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]
        out.index_add_(0, dst, values[src] * alpha[start:end].unsqueeze(1))
    return out


def _propagate_power(
    *,
    edge_index: torch.Tensor,
    values: torch.Tensor,
    num_nodes: int,
    steps: int,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    z = values
    for _ in range(max(0, int(steps))):
        z = _propagate_once(
            edge_index=edge_index,
            values=z,
            num_nodes=int(num_nodes),
            raw_edge_weight=raw_edge_weight,
            chunk_size=int(chunk_size),
        )
    return z


def smooth_logits(
    *,
    edge_index: torch.Tensor,
    logits: torch.Tensor,
    num_nodes: int,
    alpha: float,
    steps: int = 1,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
) -> LogitCorrectResult:
    z0 = _ensure_logits(logits, int(num_nodes))
    propagated = _propagate_power(
        edge_index=edge_index,
        values=z0,
        num_nodes=int(num_nodes),
        steps=int(steps),
        raw_edge_weight=raw_edge_weight,
        chunk_size=int(chunk_size),
    )
    lam = float(alpha)
    out = (1.0 - lam) * z0 + lam * propagated if int(steps) > 0 else z0
    return LogitCorrectResult(
        logits=out,
        diagnostics={
            "mode": "smooth_logits",
            "normalization": "destination_row",
            "alpha": lam,
            "steps": int(steps),
            "uses_train_labels": False,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "propagates_features": False,
        },
    )


def smooth_prob(
    *,
    edge_index: torch.Tensor,
    logits: torch.Tensor,
    num_nodes: int,
    alpha: float,
    steps: int = 1,
    temperature: float = 1.0,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
    eps: float = 1e-12,
) -> LogitCorrectResult:
    z0 = _ensure_logits(logits, int(num_nodes))
    p0 = torch.softmax(z0 / max(float(temperature), eps), dim=1)
    propagated = _propagate_power(
        edge_index=edge_index,
        values=p0,
        num_nodes=int(num_nodes),
        steps=int(steps),
        raw_edge_weight=raw_edge_weight,
        chunk_size=int(chunk_size),
    )
    lam = float(alpha)
    p = (1.0 - lam) * p0 + lam * propagated if int(steps) > 0 else p0
    p = p.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    return LogitCorrectResult(
        logits=torch.log(p),
        diagnostics={
            "mode": "smooth_prob",
            "normalization": "destination_row",
            "alpha": lam,
            "steps": int(steps),
            "temperature": float(temperature),
            "uses_train_labels": False,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "propagates_features": False,
        },
    )


def correct_error_then_smooth(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    correct_steps: int,
    correct_alpha: float,
    beta: float,
    smooth_steps: int,
    smooth_alpha: float,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
    eps: float = 1e-12,
) -> LogitCorrectResult:
    z0 = _ensure_logits(logits, int(num_nodes))
    labels = labels.to(device=z0.device, dtype=torch.long)
    train_idx = train_idx.to(device=z0.device, dtype=torch.long)
    num_classes = int(z0.shape[1])

    p0 = torch.softmax(z0, dim=1)
    error = torch.zeros_like(p0)
    if train_idx.numel() > 0:
        y_train = labels[train_idx]
        error[train_idx] = torch.nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32) - p0[train_idx]

    smoothed_error = smooth_logits(
        edge_index=edge_index,
        logits=error,
        num_nodes=int(num_nodes),
        alpha=float(correct_alpha),
        steps=int(correct_steps),
        raw_edge_weight=raw_edge_weight,
        chunk_size=int(chunk_size),
    ).logits
    p_corrected = p0 + float(beta) * smoothed_error
    p_corrected = p_corrected.clamp_min(eps)
    p_corrected = p_corrected / p_corrected.sum(dim=1, keepdim=True).clamp_min(eps)
    propagated = _propagate_power(
        edge_index=edge_index,
        values=p_corrected,
        num_nodes=int(num_nodes),
        steps=int(smooth_steps),
        raw_edge_weight=raw_edge_weight,
        chunk_size=int(chunk_size),
    )
    lam = float(smooth_alpha)
    p_final = (1.0 - lam) * p_corrected + lam * propagated if int(smooth_steps) > 0 else p_corrected
    p_final = p_final.clamp_min(eps)
    p_final = p_final / p_final.sum(dim=1, keepdim=True).clamp_min(eps)
    return LogitCorrectResult(
        logits=torch.log(p_final),
        diagnostics={
            "mode": "correct_error_then_smooth",
            "normalization": "destination_row",
            "correct_alpha": float(correct_alpha),
            "correct_steps": int(correct_steps),
            "beta": float(beta),
            "smooth_alpha": float(smooth_alpha),
            "smooth_steps": int(smooth_steps),
            "train_label_count": int(train_idx.numel()),
            "uses_train_labels": True,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "propagates_features": False,
        },
    )


def logit_correct_grid(products: bool = False) -> list[dict]:
    if products:
        alphas = [0.05, 0.1, 0.2, 0.4]
        steps = [1, 2, 4]
        temperatures = [1.0, 2.0]
    else:
        alphas = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
        steps = [1, 2, 4, 8]
        temperatures = [1.0, 2.0, 4.0]
    rows: list[dict] = []
    for space in ("logits", "probabilities"):
        for alpha in alphas:
            for step in steps:
                for temp in temperatures:
                    rows.append({"mode": "smooth_logits" if space == "logits" else "smooth_prob", "space": space, "alpha": alpha, "steps": step, "temperature": temp})
    for correct_alpha in [0.1, 0.2, 0.4, 0.6]:
        for correct_steps in [1, 2, 4]:
            for beta in [0.25, 0.5, 1.0]:
                rows.append(
                    {
                        "mode": "correct_error_then_smooth",
                        "space": "probabilities",
                        "alpha": "",
                        "steps": "",
                        "temperature": 1.0,
                        "correct_alpha": correct_alpha,
                        "correct_steps": correct_steps,
                        "beta": beta,
                    }
                )
    return rows
