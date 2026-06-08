from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved


def _positive_gate_param(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value)).clamp_min(1e-6)))


def _make_logits_mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float, bias: bool) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(in_dim, out_dim, bias=bias)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=bias),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim, bias=bias),
    )


class ShadowFusionClassifier(nn.Module):
    """Late-fusion classifier with explicit weighted relation scatter-add."""

    def __init__(
        self,
        *,
        in_channels: dict[str, int],
        out_channels: int,
        node_types: Iterable[str],
        relations: Iterable[DirectedRelation],
        target_type: str,
        block_in_channels: dict[str, int] | None = None,
        original_node_types: Iterable[str] | None = None,
        original_relations: Iterable[DirectedRelation] | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        relation_gate: bool = True,
        block_gate: bool = True,
        relation_gate_init: float = 1.0,
        block_gate_init: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.relations = list(relations)
        self.target_type = target_type
        self.block_in_channels = dict(block_in_channels or {})
        self.relation_gate_enabled = bool(relation_gate)
        self.block_gate_enabled = bool(block_gate)

        if original_node_types is not None or original_relations is not None:
            ensure_schema_preserved(
                exposed_node_types=self.node_types,
                exposed_relations=self.relations,
                original_node_types=original_node_types or self.node_types,
                original_relations=original_relations or self.relations,
            )
        if target_type not in in_channels:
            raise ValueError(f"target_type {target_type!r} is missing from in_channels")

        self.self_mlps = nn.ModuleDict(
            {
                target_type: _make_logits_mlp(
                    in_channels[target_type],
                    out_channels,
                    hidden_dim,
                    dropout,
                    bias,
                )
            }
        )
        self.relation_mlps = nn.ModuleDict(
            {
                str(relation): _make_logits_mlp(
                    in_channels[relation.source_type],
                    out_channels,
                    hidden_dim,
                    dropout,
                    bias,
                )
                for relation in self.relations
                if relation.destination_type == target_type
            }
        )
        self.block_mlps = nn.ModuleDict(
            {
                name: _make_logits_mlp(dim, out_channels, hidden_dim, dropout, bias)
                for name, dim in self.block_in_channels.items()
            }
        )
        rel_init = _positive_gate_param(relation_gate_init)
        block_init = _positive_gate_param(block_gate_init)
        self.relation_gate_params = nn.ParameterDict(
            {str(relation): nn.Parameter(rel_init.clone()) for relation in self.relations if relation.destination_type == target_type}
        )
        self.block_gate_params = nn.ParameterDict(
            {name: nn.Parameter(block_init.clone()) for name in self.block_in_channels}
        )

    def _relation_gate(self, relation: DirectedRelation) -> torch.Tensor | float:
        if not self.relation_gate_enabled:
            return 1.0
        return F.softplus(self.relation_gate_params[str(relation)])

    def _block_gate(self, name: str) -> torch.Tensor | float:
        if not self.block_gate_enabled:
            return 1.0
        return F.softplus(self.block_gate_params[name])

    def relation_gate_values(self) -> dict[str, float]:
        return self.diagnostics()["relation_gates"]

    def block_gate_values(self) -> dict[str, float]:
        return self.diagnostics()["block_gates"]

    def diagnostics(self) -> dict[str, object]:
        relation_gates = {}
        for relation in self.relations:
            if relation.destination_type != self.target_type:
                continue
            value = self._relation_gate(relation)
            relation_gates[str(relation)] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        block_gates = {}
        for name in self.block_in_channels:
            value = self._block_gate(name)
            block_gates[name] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        return {
            "final_logits_activation": "none",
            "relation_gates": relation_gates,
            "block_gates": block_gates,
        }

    def _aggregate_relation(
        self,
        relation: DirectedRelation,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        edge_chunk_size: int | None,
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
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index[0].to(source_features.device), edge_index[1].to(source_features.device)
        weight = edge_weight_dict[relation].to(device=source_features.device, dtype=source_features.dtype)
        chunk_size = edge_chunk_size or edge_index.shape[1]
        if chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        for start in range(0, edge_index.shape[1], chunk_size):
            end = min(start + chunk_size, edge_index.shape[1])
            message = source_features[src[start:end]] * weight[start:end].unsqueeze(-1)
            out.index_add_(0, dst[start:end], message)
        return out

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        block_feature_dict: dict[str, dict[str, torch.Tensor]] | None = None,
        edge_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.self_mlps[self.target_type](x_dict[self.target_type])
        for relation in self.relations:
            if relation.destination_type != self.target_type:
                continue
            msg = self._aggregate_relation(
                relation,
                x_dict,
                edge_index_dict,
                edge_weight_dict,
                edge_chunk_size=edge_chunk_size,
            )
            logits = logits + self._relation_gate(relation) * self.relation_mlps[str(relation)](msg)

        target_blocks = (block_feature_dict or {}).get(self.target_type, {})
        for name, block in target_blocks.items():
            if name not in self.block_mlps:
                continue
            logits = logits + self._block_gate(name) * self.block_mlps[name](block)
        return {self.target_type: logits}

    def _aggregate_relation_for_dst_chunk(
        self,
        relation: DirectedRelation,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        dst_start: int,
        dst_end: int,
        edge_chunk_size: int | None = None,
    ) -> torch.Tensor:
        source_features = x_dict[relation.source_type]
        out = torch.zeros(
            dst_end - dst_start,
            source_features.shape[1],
            dtype=source_features.dtype,
            device=source_features.device,
        )
        if relation not in edge_index_dict:
            return out
        edge_index = edge_index_dict[relation]
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index[0].to(source_features.device), edge_index[1].to(source_features.device)
        mask = (dst >= dst_start) & (dst < dst_end)
        if not bool(mask.any()):
            return out
        selected_src = src[mask]
        local_dst = dst[mask] - dst_start
        weight = edge_weight_dict[relation].to(device=source_features.device, dtype=source_features.dtype)[mask]
        chunk_size = edge_chunk_size or int(selected_src.numel())
        if chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        for start in range(0, int(selected_src.numel()), chunk_size):
            end = min(start + chunk_size, int(selected_src.numel()))
            out.index_add_(
                0,
                local_dst[start:end],
                source_features[selected_src[start:end]] * weight[start:end].unsqueeze(-1),
            )
        return out

    def infer_target_chunked(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[DirectedRelation, torch.Tensor],
        edge_weight_dict: dict[DirectedRelation, torch.Tensor],
        *,
        block_feature_dict: dict[str, dict[str, torch.Tensor]] | None = None,
        dst_chunk_size: int,
        edge_chunk_size: int | None = 200000,
    ) -> torch.Tensor:
        if dst_chunk_size <= 0:
            raise ValueError("dst_chunk_size must be positive")
        chunks = []
        num_target = x_dict[self.target_type].shape[0]
        target_blocks = (block_feature_dict or {}).get(self.target_type, {})
        for start in range(0, num_target, dst_chunk_size):
            end = min(start + dst_chunk_size, num_target)
            logits = self.self_mlps[self.target_type](x_dict[self.target_type][start:end])
            for relation in self.relations:
                if relation.destination_type != self.target_type:
                    continue
                msg = self._aggregate_relation_for_dst_chunk(
                    relation,
                    x_dict,
                    edge_index_dict,
                    edge_weight_dict,
                    dst_start=start,
                    dst_end=end,
                    edge_chunk_size=edge_chunk_size,
                )
                logits = logits + self._relation_gate(relation) * self.relation_mlps[str(relation)](msg)
            for name, block in target_blocks.items():
                if name not in self.block_mlps:
                    continue
                logits = logits + self._block_gate(name) * self.block_mlps[name](block[start:end])
            chunks.append(logits)
        return torch.cat(chunks, dim=0) if chunks else x_dict[self.target_type].new_zeros((0, 0))
