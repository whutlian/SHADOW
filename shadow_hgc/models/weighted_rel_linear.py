from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from shadow_hgc.data.schemas import DirectedRelation


class WeightedRelationLinearConv(nn.Module):
    """Demand-compatible weighted relation-linear layer with explicit scatter-add."""

    def __init__(
        self,
        *,
        in_channels: dict[str, int],
        out_channels: int,
        node_types: Iterable[str],
        relations: Iterable[DirectedRelation],
        activation: nn.Module | None = nn.ReLU(),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.relations = list(relations)
        self.activation = activation
        self.self_linears = nn.ModuleDict(
            {
                node_type: nn.Linear(in_channels[node_type], out_channels, bias=bias)
                for node_type in self.node_types
            }
        )
        self.relation_linears = nn.ModuleDict(
            {
                str(relation): nn.Linear(in_channels[relation.source_type], out_channels, bias=False)
                for relation in self.relations
            }
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            out[node_type] = self.self_linears[node_type](x_dict[node_type])

        for relation in self.relations:
            if relation not in edge_index_dict:
                continue
            edge_index = edge_index_dict[relation]
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            weight = edge_weight_dict[relation].to(x_dict[relation.source_type].device)
            projected = self.relation_linears[str(relation)](x_dict[relation.source_type])
            if edge_chunk_size is None:
                message = projected[src] * weight.to(projected.dtype).unsqueeze(-1)
                out[relation.destination_type].index_add_(0, dst, message)
            else:
                if edge_chunk_size <= 0:
                    raise ValueError("edge_chunk_size must be positive")
                for start in range(0, edge_index.shape[1], edge_chunk_size):
                    end = min(start + edge_chunk_size, edge_index.shape[1])
                    chunk_src = src[start:end]
                    chunk_dst = dst[start:end]
                    chunk_weight = weight[start:end].to(projected.dtype).unsqueeze(-1)
                    message = projected[chunk_src] * chunk_weight
                    out[relation.destination_type].index_add_(0, chunk_dst, message)

        if self.activation is not None:
            out = {node_type: self.activation(value) for node_type, value in out.items()}
        return out
