from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

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


class RelationMessageEncoderMLP(nn.Module):
    """One-hop weighted relation encoder followed by an MLP classifier."""

    def __init__(
        self,
        *,
        in_channels: dict[str, int],
        out_channels: int,
        node_types: Iterable[str],
        relations: Iterable[DirectedRelation],
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.relations = list(relations)
        self.dropout = nn.Dropout(dropout)
        self.self_linears = nn.ModuleDict(
            {node_type: nn.Linear(in_channels[node_type], in_channels[node_type], bias=False) for node_type in self.node_types}
        )
        self.mlps = nn.ModuleDict()
        for node_type in self.node_types:
            incoming_dim = in_channels[node_type]
            for relation in self.relations:
                if relation.destination_type == node_type:
                    incoming_dim += in_channels[relation.source_type]
            self.mlps[node_type] = nn.Sequential(
                nn.Linear(incoming_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_channels),
            )

    def _aggregate_relation_blocks(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        relation_messages: dict[str, list[torch.Tensor]] = {node_type: [] for node_type in self.node_types}
        for relation in self.relations:
            if relation not in edge_index_dict:
                continue
            edge_index = edge_index_dict[relation]
            src, dst = edge_index[0], edge_index[1]
            source_features = x_dict[relation.source_type]
            out = torch.zeros(
                x_dict[relation.destination_type].shape[0],
                source_features.shape[1],
                dtype=source_features.dtype,
                device=source_features.device,
            )
            weight = edge_weight_dict[relation].to(source_features.device)
            if edge_chunk_size is None:
                out.index_add_(0, dst, source_features[src] * weight.to(source_features.dtype).unsqueeze(-1))
            else:
                for start in range(0, edge_index.shape[1], edge_chunk_size):
                    end = min(start + edge_chunk_size, edge_index.shape[1])
                    chunk_src = src[start:end]
                    chunk_dst = dst[start:end]
                    chunk_weight = weight[start:end].to(source_features.dtype).unsqueeze(-1)
                    out.index_add_(0, chunk_dst, source_features[chunk_src] * chunk_weight)
            relation_messages[relation.destination_type].append(out)
        return relation_messages

    def encode_messages(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        relation_blocks = self._aggregate_relation_blocks(
            x_dict,
            edge_index_dict,
            edge_weight_dict,
            edge_chunk_size=edge_chunk_size,
        )
        relation_messages: dict[str, list[torch.Tensor]] = {}
        for node_type in self.node_types:
            relation_messages[node_type] = [self.self_linears[node_type](x_dict[node_type]), *relation_blocks[node_type]]
        return {node_type: torch.cat(blocks, dim=1) for node_type, blocks in relation_messages.items()}

    def _aggregate_one_relation(
        self,
        relation: DirectedRelation,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int,
    ) -> torch.Tensor:
        source_features = x_dict[relation.source_type]
        out = torch.zeros(
            x_dict[relation.destination_type].shape[0],
            source_features.shape[1],
            dtype=source_features.dtype,
            device=source_features.device,
        )
        if relation not in edge_index_dict:
            return out
        edge_index = edge_index_dict[relation]
        src, dst = edge_index[0], edge_index[1]
        weight = edge_weight_dict[relation].to(source_features.device)
        for start in range(0, edge_index.shape[1], edge_chunk_size):
            end = min(start + edge_chunk_size, edge_index.shape[1])
            chunk_src = src[start:end]
            chunk_dst = dst[start:end]
            chunk_weight = weight[start:end].to(source_features.dtype).unsqueeze(-1)
            out.index_add_(0, chunk_dst, source_features[chunk_src] * chunk_weight)
        return out

    def _forward_memory_efficient(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            first: nn.Linear = self.mlps[node_type][0]
            second: nn.Linear = self.mlps[node_type][3]
            hidden = x_dict[node_type].new_zeros((x_dict[node_type].shape[0], first.out_features))
            if first.bias is not None:
                hidden += first.bias

            offset = 0
            self_dim = self.self_linears[node_type].out_features
            self_weight = first.weight[:, offset : offset + self_dim]
            for start in range(0, x_dict[node_type].shape[0], edge_chunk_size):
                end = min(start + edge_chunk_size, x_dict[node_type].shape[0])
                self_block = self.self_linears[node_type](x_dict[node_type][start:end])
                hidden[start:end] += F.linear(self_block, self_weight)
            offset += self_dim

            for relation in self.relations:
                if relation.destination_type != node_type:
                    continue
                rel_dim = x_dict[relation.source_type].shape[1]
                rel_weight = first.weight[:, offset : offset + rel_dim]
                rel_block = self._aggregate_one_relation(
                    relation,
                    x_dict,
                    edge_index_dict,
                    edge_weight_dict,
                    edge_chunk_size=edge_chunk_size,
                )
                hidden += F.linear(rel_block, rel_weight)
                offset += rel_dim

            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
            chunks = []
            for start in range(0, hidden.shape[0], edge_chunk_size):
                end = min(start + edge_chunk_size, hidden.shape[0])
                chunks.append(second(hidden[start:end]))
            out[node_type] = torch.cat(chunks, dim=0) if chunks else second(hidden)
        return out

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if edge_chunk_size is not None:
            return self._forward_memory_efficient(
                x_dict,
                edge_index_dict,
                edge_weight_dict,
                edge_chunk_size=edge_chunk_size,
            )
        encoded = self.encode_messages(x_dict, edge_index_dict, edge_weight_dict, edge_chunk_size=edge_chunk_size)
        return {node_type: self.mlps[node_type](features) for node_type, features in encoded.items()}
