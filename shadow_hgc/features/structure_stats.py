from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.demand.normalize import destination_degrees
from shadow_hgc.features.degree import encode_degree_vector


def compute_structure_stats_block(
    *,
    edge_index_by_relation: dict[DirectedRelation, torch.Tensor],
    relations: list[DirectedRelation],
    num_target_nodes: int,
    target_rows: torch.Tensor,
    scap_diagnostics: dict[str, dict] | None = None,
) -> tuple[torch.Tensor, dict]:
    blocks: list[torch.Tensor] = []
    dims: dict[str, int] = {}
    for relation in relations:
        degree = destination_degrees(edge_index_by_relation[relation], int(num_target_nodes))
        encoded = encode_degree_vector(degree)[target_rows.to(torch.long)]
        blocks.append(encoded)
        dims[str(relation)] = int(encoded.shape[1])
    if not blocks:
        out = torch.empty(target_rows.numel(), 0)
    else:
        out = torch.cat(blocks, dim=1).to(torch.float32)
    diagnostics = {
        "structure_stats_dim": int(out.shape[1]),
        "structure_relation_dims": dims,
        "scap_stats_attached": bool(scap_diagnostics),
    }
    return out, diagnostics
