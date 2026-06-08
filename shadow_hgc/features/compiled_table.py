from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from shadow_hgc.data.schemas import DirectedRelation


@dataclass(frozen=True)
class CompiledDemandBlock:
    name: str
    kind: Literal["self", "feature_demand", "label_affinity", "degree"]
    relation: DirectedRelation | None
    dim: int
    normalize: str


@dataclass(frozen=True)
class CompiledDemandSchema:
    blocks: list[CompiledDemandBlock]
    total_dim: int


def _relation_to_json(relation: DirectedRelation | None) -> dict | None:
    if relation is None:
        return None
    return {
        "source_type": relation.source_type,
        "relation_name": relation.relation_name,
        "destination_type": relation.destination_type,
    }


def schema_to_dict(schema: CompiledDemandSchema) -> dict:
    return {
        "total_dim": int(schema.total_dim),
        "blocks": [
            {
                **asdict(block),
                "relation": _relation_to_json(block.relation),
                "dim": int(block.dim),
            }
            for block in schema.blocks
        ],
    }


def schema_to_json(schema: CompiledDemandSchema) -> str:
    return json.dumps(schema_to_dict(schema), sort_keys=True)


def compile_demand_table(
    blocks: list[tuple[CompiledDemandBlock, torch.Tensor]],
) -> tuple[torch.Tensor, CompiledDemandSchema]:
    if not blocks:
        raise ValueError("at least one compiled demand block is required")
    num_rows = int(blocks[0][1].shape[0])
    tensors: list[torch.Tensor] = []
    schema_blocks: list[CompiledDemandBlock] = []
    for block, tensor in blocks:
        if tensor.ndim != 2:
            raise ValueError(f"{block.name}: tensor must be rank-2")
        if int(tensor.shape[0]) != num_rows:
            raise ValueError(f"{block.name}: row count mismatch")
        if int(tensor.shape[1]) != int(block.dim):
            raise ValueError(f"{block.name}: expected dim {block.dim}, got {tensor.shape[1]}")
        tensors.append(tensor.to(torch.float32))
        schema_blocks.append(block)
    table = torch.cat(tensors, dim=1)
    total_dim = sum(int(block.dim) for block in schema_blocks)
    if int(table.shape[1]) != total_dim:
        raise ValueError("compiled table total dimension mismatch")
    return table, CompiledDemandSchema(blocks=schema_blocks, total_dim=total_dim)


def block_slices(schema: CompiledDemandSchema) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for block in schema.blocks:
        end = start + int(block.dim)
        slices[block.name] = slice(start, end)
        start = end
    return slices
