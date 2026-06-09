from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass
class MetaPathFeatureBlocks:
    blocks: dict[str, torch.Tensor]
    exposed_relations: list[DirectedRelation]
    skipped_blocks: list[str]


def _type_letter(node_type: str) -> str:
    return node_type[:1].upper()


def two_hop_block_name(target_type: str, source_type: str) -> str:
    return f"{_type_letter(target_type)}{_type_letter(source_type)}{_type_letter(target_type)}"


def _target_source_target_features(
    edge_index: torch.Tensor,
    *,
    target_features: torch.Tensor,
    num_source_nodes: int,
    num_target_nodes: int,
) -> torch.Tensor:
    device = target_features.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.numel() == 0:
        return torch.zeros(num_target_nodes, target_features.shape[1], dtype=target_features.dtype, device=device)

    src = edge_index[0]
    dst = edge_index[1]
    reverse_edge_index = torch.stack([dst, src], dim=0)
    reverse_alpha = destination_row_normalize(reverse_edge_index, num_source_nodes).to(device=device, dtype=target_features.dtype)
    source_summary = torch.zeros(num_source_nodes, target_features.shape[1], dtype=target_features.dtype, device=device)
    source_summary.index_add_(0, src, target_features[dst] * reverse_alpha.unsqueeze(1))

    forward_alpha = destination_row_normalize(edge_index, num_target_nodes).to(device=device, dtype=target_features.dtype)
    out = torch.zeros(num_target_nodes, target_features.shape[1], dtype=target_features.dtype, device=device)
    out.index_add_(0, dst, source_summary[src] * forward_alpha.unsqueeze(1))
    return out


def compute_metapath_feature_blocks(
    *,
    edge_index: dict[DirectedRelation, torch.Tensor],
    relations: list[DirectedRelation],
    target_type: str,
    target_features: torch.Tensor,
    num_nodes: dict[str, int],
    requested_blocks: list[str] | None = None,
) -> MetaPathFeatureBlocks:
    """Compute target-source-target feature blocks without exposing meta-path edges."""

    requested = None if requested_blocks is None else [name.upper() for name in requested_blocks]
    blocks: dict[str, torch.Tensor] = {}
    exposed: list[DirectedRelation] = []
    available_names: set[str] = set()
    for relation in relations:
        if relation.destination_type != target_type or relation.source_type == target_type:
            continue
        name = two_hop_block_name(target_type, relation.source_type)
        available_names.add(name)
        if requested is not None and name not in requested:
            continue
        blocks[name] = _target_source_target_features(
            edge_index[relation],
            target_features=target_features,
            num_source_nodes=int(num_nodes[relation.source_type]),
            num_target_nodes=int(num_nodes[target_type]),
        )
        exposed.append(relation)

    skipped = [] if requested is None else [name for name in requested if name not in available_names]
    return MetaPathFeatureBlocks(blocks=blocks, exposed_relations=exposed, skipped_blocks=skipped)
