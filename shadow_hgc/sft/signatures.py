from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from shadow_hgc.models.sft_teacher import _contains_forbidden_logit_name


@dataclass(frozen=True)
class SFTSignatureResult:
    signature: torch.Tensor
    block_names: list[str]
    block_dims: dict[str, int]
    train_rows: torch.Tensor
    stats: dict[str, dict[str, object]]
    uses_logits_as_input: bool = False
    uses_teacher_logits: bool = False


def _fit_standardize(block: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    block = block.to(torch.float32)
    train = block[rows]
    mean = train.mean(dim=0, keepdim=True) if train.numel() else torch.zeros(1, block.shape[1], dtype=torch.float32)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6) if train.numel() else torch.ones(1, block.shape[1], dtype=torch.float32)
    return (block - mean) / std, {
        "fit_scope": "train_target_rows",
        "frozen": True,
        "mean_shape": [int(value) for value in mean.shape],
        "std_min": float(std.min().item()) if std.numel() else 1.0,
    }


def build_sft_signature(
    blocks: Mapping[str, torch.Tensor],
    *,
    train_rows: torch.Tensor | Iterable[int],
    selected_blocks: Iterable[str] | None = None,
    block_norm: bool = True,
) -> SFTSignatureResult:
    if not blocks:
        raise ValueError("at least one SFT block is required")
    rows = torch.as_tensor(list(train_rows) if not torch.is_tensor(train_rows) else train_rows, dtype=torch.long)
    names = [str(name) for name in (selected_blocks or blocks.keys())]
    forbidden = [name for name in names if _contains_forbidden_logit_name(name)]
    if forbidden:
        raise ValueError(f"logits are forbidden in T23 SFT signatures: {forbidden}")
    pieces: list[torch.Tensor] = []
    dims: dict[str, int] = {}
    stats: dict[str, dict[str, object]] = {}
    for name in names:
        if name not in blocks:
            raise ValueError(f"missing SFT signature block {name}")
        block = blocks[name].to(torch.float32)
        if block.ndim != 2:
            raise ValueError(f"SFT signature block must be 2D: {name}")
        dims[name] = int(block.shape[1])
        if block_norm:
            block, stats[name] = _fit_standardize(block, rows)
        else:
            stats[name] = {"fit_scope": "disabled", "frozen": True}
        pieces.append(block)
    signature = torch.cat(pieces, dim=1) if pieces else torch.zeros(0, 0)
    return SFTSignatureResult(
        signature=signature,
        block_names=names,
        block_dims=dims,
        train_rows=rows,
        stats=stats,
    )
