from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.demand.normalize import destination_degrees

_NUM_BUCKETS = 9


def degree_encoding_dim() -> int:
    return 1 + _NUM_BUCKETS + 1


def bucket_ids(degree: torch.Tensor) -> torch.Tensor:
    buckets = torch.empty_like(degree, dtype=torch.long)
    buckets[degree == 0] = 0
    buckets[degree == 1] = 1
    buckets[degree == 2] = 2
    buckets[(degree >= 3) & (degree <= 4)] = 3
    buckets[(degree >= 5) & (degree <= 8)] = 4
    buckets[(degree >= 9) & (degree <= 16)] = 5
    buckets[(degree >= 17) & (degree <= 32)] = 6
    buckets[(degree >= 33) & (degree <= 64)] = 7
    buckets[degree > 64] = 8
    return buckets


def encode_degree_vector(degree: torch.Tensor) -> torch.Tensor:
    degree = degree.to(torch.long)
    out = torch.zeros(degree.numel(), degree_encoding_dim(), dtype=torch.float32, device=degree.device)
    out[:, 0] = torch.log1p(degree.to(torch.float32))
    out[torch.arange(degree.numel(), device=degree.device), 1 + bucket_ids(degree)] = 1.0
    out[:, -1] = (degree == 0).to(torch.float32)
    return out


def encode_target_degrees(
    degree_by_relation: dict[DirectedRelation, torch.Tensor],
    relations: list[DirectedRelation],
) -> torch.Tensor:
    blocks = [encode_degree_vector(degree_by_relation[relation]) for relation in relations]
    return torch.cat(blocks, dim=1) if blocks else torch.empty(0, 0)


def compute_degree_features(
    edge_index_by_relation: dict[DirectedRelation, torch.Tensor],
    relations: list[DirectedRelation],
    *,
    num_target_nodes: int,
) -> tuple[dict[DirectedRelation, torch.Tensor], torch.Tensor]:
    degree_by_relation = {
        relation: destination_degrees(edge_index_by_relation[relation], num_target_nodes)
        for relation in relations
    }
    return degree_by_relation, encode_target_degrees(degree_by_relation, relations)
