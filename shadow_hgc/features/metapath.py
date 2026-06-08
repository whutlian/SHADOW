from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation


@dataclass
class MetapathFeatureResult:
    features: torch.Tensor
    path_names: list[str]
    exposed_relations: list[DirectedRelation]


def _source_target_mean(
    edge_index: torch.Tensor,
    psi_target: torch.Tensor,
    *,
    num_source_nodes: int,
) -> torch.Tensor:
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    out = torch.zeros(num_source_nodes, psi_target.shape[1], dtype=psi_target.dtype, device=psi_target.device)
    count = torch.zeros(num_source_nodes, 1, dtype=psi_target.dtype, device=psi_target.device)
    if edge_index.numel() > 0:
        out.index_add_(0, src.to(psi_target.device), psi_target[dst.to(psi_target.device)])
        count.index_add_(0, src.to(psi_target.device), torch.ones(src.numel(), 1, dtype=psi_target.dtype, device=psi_target.device))
    return out / count.clamp_min(1.0)


def metapath_target_features(
    *,
    edge_index: dict[DirectedRelation, torch.Tensor],
    relations: list[DirectedRelation],
    target_type: str,
    psi_target: torch.Tensor,
    num_nodes: dict[str, int],
) -> MetapathFeatureResult:
    """Compute tau-sigma-tau target features without materializing meta-path edge types."""

    pieces: list[torch.Tensor] = []
    names: list[str] = []
    exposed: list[DirectedRelation] = []
    for relation in relations:
        if relation.destination_type != target_type or relation.source_type == target_type:
            continue
        rel_edges = edge_index[relation]
        source_mean = _source_target_mean(
            rel_edges,
            psi_target,
            num_source_nodes=int(num_nodes[relation.source_type]),
        )
        src = rel_edges[0].to(torch.long)
        dst = rel_edges[1].to(torch.long)
        target_out = torch.zeros_like(psi_target)
        count = torch.zeros(psi_target.shape[0], 1, dtype=psi_target.dtype, device=psi_target.device)
        if rel_edges.numel() > 0:
            target_out.index_add_(0, dst.to(psi_target.device), source_mean[src.to(psi_target.device)])
            count.index_add_(0, dst.to(psi_target.device), torch.ones(dst.numel(), 1, dtype=psi_target.dtype, device=psi_target.device))
        pieces.append(target_out / count.clamp_min(1.0))
        names.append(f"{target_type}-{relation.source_type}-{target_type}")
        exposed.append(relation)
    if not pieces:
        return MetapathFeatureResult(torch.empty(psi_target.shape[0], 0, dtype=psi_target.dtype, device=psi_target.device), [], [])
    return MetapathFeatureResult(torch.cat(pieces, dim=1), names, exposed)
