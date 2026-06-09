from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass
class TwoHopLADResult:
    blocks: dict[str, torch.Tensor]
    diagnostics: dict


def _propagate(block: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, block.shape[1], dtype=block.dtype, device=block.device)
    if edge_index.numel() == 0:
        return out
    edge_index = edge_index.to(device=block.device, dtype=torch.long)
    alpha = destination_row_normalize(edge_index, int(num_nodes)).to(device=block.device, dtype=block.dtype)
    out.index_add_(0, edge_index[1], block[edge_index[0]] * alpha.unsqueeze(1))
    return out


def compute_two_hop_lad(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    steps: int = 2,
    smoothing: float = 1e-4,
) -> TwoHopLADResult:
    labels = train_labels.to(torch.long)
    train_mask = train_target_mask.to(torch.bool)
    y = torch.full_like(labels, -1)
    y[train_mask] = labels[train_mask]
    valid = (y >= 0) & (y < int(num_classes))
    block = torch.zeros(int(num_nodes), int(num_classes), dtype=torch.float32, device=edge_index.device)
    if bool(valid.any()):
        block[valid.to(block.device), y[valid].to(block.device)] = 1.0
    blocks: dict[str, torch.Tensor] = {}
    current = block
    for step in range(1, int(steps) + 1):
        current = _propagate(current, edge_index, int(num_nodes))
        normalized = current + float(smoothing)
        normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1e-12)
        blocks[f"P{step}"] = normalized
    return TwoHopLADResult(
        blocks=blocks,
        diagnostics={
            "lad_num_classes": int(num_classes),
            "two_hop_lad_steps": int(steps),
            "two_hop_lad_normalize": "row",
            "two_hop_lad_smoothing": float(smoothing),
            "uses_train_labels_only": True,
            "uses_feature_diffusion": False,
            "complexity": "O(E*C)",
        },
    )

