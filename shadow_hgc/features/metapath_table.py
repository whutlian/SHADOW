from __future__ import annotations

import time
from typing import Mapping

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.demand.normalize import destination_row_normalize


def _aggregate_relation(
    *,
    relation: DirectedRelation,
    edge_index: torch.Tensor,
    source_table: torch.Tensor,
    num_dst_nodes: int,
    chunk_size: int,
) -> torch.Tensor:
    device = source_table.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    out = torch.zeros(int(num_dst_nodes), int(source_table.shape[1]), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return out
    alpha = destination_row_normalize(edge_index, int(num_dst_nodes)).to(device=device, dtype=torch.float32)
    for start in range(0, int(edge_index.shape[1]), int(chunk_size)):
        end = min(int(edge_index.shape[1]), start + int(chunk_size))
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]
        out.index_add_(0, dst, source_table[src] * alpha[start:end].unsqueeze(1))
    return out


def compute_metapath_feature(
    *,
    path_schema: list[DirectedRelation],
    target_type: str,
    feature_provider: Mapping[str, torch.Tensor],
    edge_store: Mapping[DirectedRelation, torch.Tensor],
    num_nodes: Mapping[str, int],
    target_rows: torch.Tensor,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, dict]:
    if not path_schema:
        raise ValueError("path_schema must contain at least one directed relation")
    started = time.perf_counter()
    source_type = path_schema[0].source_type
    if source_type not in feature_provider:
        raise KeyError(f"missing feature table for path source type {source_type}")
    current_type = source_type
    current = feature_provider[source_type].to(torch.float32)
    max_chunk = 0
    for relation in path_schema:
        if relation.source_type != current_type:
            raise ValueError(f"path relation {relation} does not continue from {current_type}")
        edge_index = edge_store[relation]
        max_chunk = max(max_chunk, min(int(edge_index.shape[1]), int(chunk_size)))
        current = _aggregate_relation(
            relation=relation,
            edge_index=edge_index,
            source_table=current,
            num_dst_nodes=int(num_nodes[relation.destination_type]),
            chunk_size=int(chunk_size),
        )
        current_type = relation.destination_type
    if current_type != target_type:
        raise ValueError(f"path ends at {current_type}, expected target type {target_type}")
    rows = target_rows.to(dtype=torch.long, device=current.device)
    block = current[rows]
    diagnostics = {
        "path": "->".join([path_schema[0].source_type, *[rel.destination_type for rel in path_schema]]),
        "path_length": int(len(path_schema)),
        "metapath_block_dim": int(block.shape[1]),
        "metapath_cache_bytes": int(block.numel() * block.element_size()),
        "metapath_precompute_time_s": float(time.perf_counter() - started),
        "normalization": "destination_row",
        "no_dense_adjacency": True,
        "materialized_dense_adjacency": False,
        "max_edge_chunk_size": int(max_chunk),
    }
    return block, diagnostics
