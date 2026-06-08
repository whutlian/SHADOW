from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass
class DiffusionFeatureResult:
    features: torch.Tensor
    block_names: list[str]


def _diffuse_once(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, x.shape[1], dtype=x.dtype, device=x.device)
    alpha = destination_row_normalize(edge_index, num_nodes).to(device=x.device, dtype=x.dtype)
    src = edge_index[0].to(x.device)
    dst = edge_index[1].to(x.device)
    out = torch.zeros(num_nodes, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, dst, x[src] * alpha.unsqueeze(-1))
    return out


def diffusion_target_features(
    x: torch.Tensor,
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: tuple[int, ...] | list[int] = (1,),
    include_highpass: bool = False,
) -> DiffusionFeatureResult:
    """Build deterministic destination-row-normalized target diffusion blocks."""

    requested = sorted({int(step) for step in steps if int(step) > 0})
    current = x
    blocks: dict[int, torch.Tensor] = {}
    for step in range(1, (max(requested) if requested else 0) + 1):
        current = _diffuse_once(current, edge_index, num_nodes)
        if step in requested:
            blocks[step] = current
    pieces = [blocks[step] for step in requested]
    names = [f"X{step}" for step in requested]
    if include_highpass:
        x1 = blocks.get(1)
        if x1 is None:
            x1 = _diffuse_once(x, edge_index, num_nodes)
        pieces.append(x - x1)
        names.append("Xhp")
    if not pieces:
        return DiffusionFeatureResult(torch.empty(num_nodes, 0, dtype=x.dtype, device=x.device), [])
    return DiffusionFeatureResult(torch.cat(pieces, dim=1), names)
