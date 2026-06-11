from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class T28CorrectSmoothResult:
    logits_or_probs: torch.Tensor
    diagnostics: dict[str, Any]


def _as_edge_chunks(edge_index_or_stream: torch.Tensor | Iterable[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(edge_index_or_stream, torch.Tensor):
        if edge_index_or_stream.ndim != 2 or edge_index_or_stream.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        return [edge_index_or_stream.to(torch.long)]
    chunks: list[torch.Tensor] = []
    for chunk in edge_index_or_stream:
        if chunk.ndim != 2 or chunk.shape[0] != 2:
            raise ValueError("edge stream chunks must have shape [2, E]")
        chunks.append(chunk.to(torch.long))
    return chunks


def _to_probabilities(logits_or_probs: torch.Tensor, eps: float) -> torch.Tensor:
    x = logits_or_probs.to(torch.float32)
    row_sum = x.sum(dim=1, keepdim=True)
    looks_like_probs = bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4))
    if looks_like_probs:
        return x.clamp_min(eps) / row_sum.clamp_min(eps)
    return torch.softmax(x, dim=1).clamp_min(eps)


def _propagate_once(values: torch.Tensor, chunks: list[torch.Tensor], num_nodes: int) -> torch.Tensor:
    out = torch.zeros((int(num_nodes), int(values.shape[1])), dtype=values.dtype, device=values.device)
    for edge_index in chunks:
        if edge_index.numel() == 0:
            continue
        edge_index = edge_index.to(device=values.device, dtype=torch.long)
        alpha = destination_row_normalize(edge_index, int(num_nodes)).to(device=values.device, dtype=values.dtype)
        out.index_add_(0, edge_index[1], values[edge_index[0]] * alpha.unsqueeze(1))
    return out


def _propagate(values: torch.Tensor, chunks: list[torch.Tensor], num_nodes: int, steps: int) -> torch.Tensor:
    z = values
    for _ in range(max(0, int(steps))):
        z = _propagate_once(z, chunks, int(num_nodes))
    return z


def correct_and_smooth(
    logits_or_probs: torch.Tensor,
    y_train: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    edge_index_or_stream: torch.Tensor | Iterable[torch.Tensor],
    *,
    num_classes: int,
    correction_alpha: float,
    smoothing_alpha: float,
    num_correction_steps: int,
    num_smoothing_steps: int,
    autoscale: bool = True,
    normalize: str = "dst_row",
    eps: float = 1e-12,
) -> T28CorrectSmoothResult:
    """Sparse T28 Correct-and-Smooth postprocess using only train labels as inputs."""

    if normalize != "dst_row":
        raise ValueError("T28 correct_and_smooth only supports normalize='dst_row'")
    probs = _to_probabilities(logits_or_probs, eps)
    if probs.shape[1] != int(num_classes):
        raise ValueError("num_classes must match logits_or_probs.shape[1]")
    num_nodes = int(probs.shape[0])
    labels = y_train.to(device=probs.device, dtype=torch.long)
    train_idx = train_idx.to(device=probs.device, dtype=torch.long)
    _ = valid_idx.numel()
    _ = test_idx.numel()
    chunks = _as_edge_chunks(edge_index_or_stream)

    residual = torch.zeros_like(probs)
    if train_idx.numel() > 0:
        one_hot = torch.nn.functional.one_hot(labels[train_idx].clamp(0, int(num_classes) - 1), num_classes=int(num_classes)).to(probs.dtype)
        residual[train_idx] = one_hot - probs[train_idx]
    propagated_residual = _propagate(residual, chunks, num_nodes, int(num_correction_steps))
    if autoscale and train_idx.numel() > 0:
        base_norm = residual[train_idx].norm(dim=1).mean().clamp_min(float(eps))
        propagated_norm = propagated_residual[train_idx].norm(dim=1).mean().clamp_min(float(eps))
        propagated_residual = propagated_residual * (base_norm / propagated_norm).clamp(max=100.0)
    corrected = (probs + float(correction_alpha) * propagated_residual).clamp_min(float(eps))
    corrected = corrected / corrected.sum(dim=1, keepdim=True).clamp_min(float(eps))

    smoothed = corrected
    for _step in range(max(0, int(num_smoothing_steps))):
        propagated = _propagate_once(smoothed, chunks, num_nodes)
        smoothed = ((1.0 - float(smoothing_alpha)) * corrected + float(smoothing_alpha) * propagated).clamp_min(float(eps))
        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(float(eps))

    return T28CorrectSmoothResult(
        logits_or_probs=smoothed,
        diagnostics={
            "uses_cns_postprocess": True,
            "uses_train_labels": True,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "uses_teacher_logits_for_condensation": False,
            "creates_dense_adjacency": False,
            "normalization": "dst_row",
            "cns_correction_alpha": float(correction_alpha),
            "cns_smoothing_alpha": float(smoothing_alpha),
            "cns_correction_steps": int(num_correction_steps),
            "cns_smoothing_steps": int(num_smoothing_steps),
            "cns_autoscale": bool(autoscale),
            "edge_chunks": len(chunks),
        },
    )
