from __future__ import annotations

from typing import Any, Mapping

import torch

from shadow_hgc.preprop.chunked_spmm import chunked_destination_row_spmm
from shadow_hgc.preprop.filter_bank import compute_label_reuse_blocks


def _relation_value(blocks: Mapping[str, torch.Tensor], step: int, relation_name: str, num_nodes: int, num_classes: int) -> torch.Tensor:
    key = f"Y{step}_{relation_name}"
    if key in blocks:
        return blocks[key]
    return torch.zeros(int(num_nodes), int(num_classes), dtype=torch.float32)


def compute_label_reuse_v2_blocks(
    *,
    relation_blocks: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    train_target_ids: torch.Tensor,
    num_target_nodes: int,
    num_classes: int,
    edge_chunk_size: int = 1_000_000,
    prior_centering: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build T23 train-label-only label reuse blocks.

    `labels` may contain labels for every target row, but only `train_target_ids`
    are read into the seed one-hot matrix. Validation/test labels are never
    materialized as label sources.
    """

    labels = labels.to(torch.long).cpu()
    train_rows = train_target_ids.to(torch.long).cpu()
    y0 = torch.zeros(int(num_target_nodes), int(num_classes), dtype=torch.float32)
    if train_rows.numel() > 0:
        y0[train_rows, labels[train_rows]] = 1.0
    base, diagnostics = compute_label_reuse_blocks(
        relation_blocks=relation_blocks,
        labels=labels,
        train_target_ids=train_rows,
        num_target_nodes=int(num_target_nodes),
        num_classes=int(num_classes),
        steps=(1, 2, 3, 4),
        prior_centering=bool(prior_centering),
        edge_chunk_size=int(edge_chunk_size),
    )
    out: dict[str, torch.Tensor] = {"Y0_train_masked": y0}
    out.update(base)
    rel_names = sorted(str(name) for name in relation_blocks)
    if rel_names:
        for step in (1, 2, 3, 4):
            pieces = [_relation_value(base, step, name, num_target_nodes, num_classes) for name in rel_names]
            out[f"Y{step}_mix"] = sum(pieces) / max(1, len(pieces))
    else:
        for step in (1, 2, 3, 4):
            out[f"Y{step}_mix"] = torch.zeros(int(num_target_nodes), int(num_classes), dtype=torch.float32)
    out["Yres1_mix"] = y0 - out["Y1_mix"]
    if prior_centering:
        hist = torch.bincount(labels[train_rows], minlength=int(num_classes)).to(torch.float32)
        prior = hist / hist.sum().clamp_min(1.0)
        for name in list(out):
            if name.startswith("Y") and out[name].ndim == 2 and int(out[name].shape[1]) == int(num_classes):
                out[f"{name}_prior_centered"] = out[name] - prior.view(1, -1)
    diagnostics = {
        **diagnostics,
        "label_reuse_version": "v2",
        "has_y0_train_masked": True,
        "has_y4": True,
        "has_yres1": True,
        "uses_valid_labels": False,
        "uses_test_labels": False,
        "label_block_dims": {name: int(block.shape[1]) for name, block in out.items()},
        "label_block_cache_bytes": sum(int(block.numel() * block.element_size()) for block in out.values()),
    }
    return out, diagnostics


def apply_label_dropout(block: torch.Tensor, *, dropout: float, training: bool, generator: torch.Generator | None = None) -> torch.Tensor:
    if not training or float(dropout) <= 0.0:
        return block
    keep_prob = max(1e-12, 1.0 - float(dropout))
    mask = (torch.rand(block.shape, device=block.device, generator=generator) < keep_prob).to(block.dtype)
    return block * mask / keep_prob
