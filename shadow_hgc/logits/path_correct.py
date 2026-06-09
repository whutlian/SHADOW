from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class PathStep:
    edge_index: torch.Tensor
    num_src: int
    num_dst: int
    name: str


@dataclass(frozen=True)
class PathLogitCorrectResult:
    logits: torch.Tensor
    diagnostics: dict[str, Any]


def _step(values: torch.Tensor, step: PathStep) -> torch.Tensor:
    edge_index = step.edge_index.to(device=values.device, dtype=torch.long)
    out = torch.zeros(int(step.num_dst), int(values.shape[1]), dtype=torch.float32, device=values.device)
    if edge_index.numel() == 0:
        return out
    alpha = destination_row_normalize(edge_index, int(step.num_dst)).to(device=values.device, dtype=torch.float32)
    out.index_add_(0, edge_index[1], values[edge_index[0]] * alpha.unsqueeze(1))
    return out


def apply_path_logit_correct(
    *,
    base_logits: torch.Tensor,
    steps: list[PathStep],
    alpha: float,
    space: str = "prob",
    eps: float = 1e-12,
) -> PathLogitCorrectResult:
    if space not in {"prob", "logit"}:
        raise ValueError("space must be prob or logit")
    base = torch.softmax(base_logits.to(torch.float32), dim=1) if space == "prob" else base_logits.to(torch.float32)
    z = base
    for step in steps:
        z = _step(z, step)
    mixed = (1.0 - float(alpha)) * base + float(alpha) * z
    if space == "prob":
        mixed = mixed.clamp_min(eps)
        mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(eps)
        logits = torch.log(mixed.clamp_min(eps))
    else:
        logits = mixed
    return PathLogitCorrectResult(
        logits=logits,
        diagnostics={
            "uses_dense_path_adjacency": False,
            "exposes_metapath_edge_type": False,
            "path_steps": [step.name for step in steps],
            "space": space,
            "alpha": float(alpha),
        },
    )
