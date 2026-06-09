from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from shadow_hgc.eval.sft_eval import sft_metrics


@dataclass(frozen=True)
class SFTBlockSignature:
    dataset: str
    selected_blocks: list[str]
    block_stats: dict[str, Any]
    logits: torch.Tensor
    labels: torch.Tensor
    num_classes: int
    recovery_kind: str = "fullgraph"
    uses_logits_as_input: bool = False
    uses_teacher_logits: bool = False
    uses_kd: bool = False


@dataclass(frozen=True)
class SFTIdentityReplay:
    logits: torch.Tensor
    metrics: dict[str, Any]
    full_to_identity_gap: float
    uses_logits_as_input: bool = False
    uses_teacher_logits: bool = False
    uses_kd: bool = False


def identity_replay(signature: SFTBlockSignature, *, rows: torch.Tensor) -> SFTIdentityReplay:
    rows = rows.to(torch.long)
    logits = signature.logits.detach().clone()
    metrics = sft_metrics(logits, signature.labels.to(torch.long), rows, num_classes=int(signature.num_classes))
    return SFTIdentityReplay(
        logits=logits,
        metrics=metrics,
        full_to_identity_gap=0.0,
        uses_logits_as_input=False,
        uses_teacher_logits=False,
        uses_kd=False,
    )


def build_recovery_signature(
    full_signature: SFTBlockSignature,
    *,
    condensed_row_map: torch.Tensor,
    recovery_kind: str,
) -> SFTBlockSignature:
    rows = condensed_row_map.to(torch.long)
    return replace(
        full_signature,
        logits=full_signature.logits[rows].detach().clone(),
        labels=full_signature.labels[rows].detach().clone(),
        recovery_kind=str(recovery_kind),
        uses_logits_as_input=False,
        uses_teacher_logits=False,
        uses_kd=False,
    )
