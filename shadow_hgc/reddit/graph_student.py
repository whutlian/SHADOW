from __future__ import annotations

import torch
from torch import nn


class WeightedGraphStudent(nn.Module):
    """Small weighted graph student for condensed Reddit graphs.

    The layer rule is explicit weighted scatter-add. The class intentionally
    avoids library GCN/GraphSAGE layers so pre-normalized edge weights are not
    normalized a second time.
    """

    uses_library_normalization = False

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        layers: int = 2,
        model_type: str = "weighted_gcn",
        residual: bool = True,
        norm: str = "layernorm",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(layers) <= 0:
            raise ValueError("layers must be positive")
        if model_type not in {"weighted_gcn", "weighted_graphsage", "weighted_sgc", "mlp_table_fallback"}:
            raise ValueError("unsupported graph student model_type")
        self.model_type = model_type
        self.residual = bool(residual)
        self.dropout = nn.Dropout(float(dropout))
        dims = [int(input_dim)] + [int(hidden_dim)] * int(layers)
        self.layers = nn.ModuleList()
        for i in range(int(layers)):
            in_dim = dims[i] * 2 if model_type == "weighted_graphsage" else dims[i]
            self.layers.append(nn.Linear(in_dim, dims[i + 1]))
        self.norms = nn.ModuleList([self._make_norm(norm, int(hidden_dim)) for _ in range(int(layers))])
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))

    @staticmethod
    def _make_norm(norm: str, dim: int) -> nn.Module:
        if norm == "layernorm":
            return nn.LayerNorm(dim)
        if norm == "batchnorm":
            return nn.BatchNorm1d(dim)
        if norm in {"none", ""}:
            return nn.Identity()
        raise ValueError("norm must be layernorm, batchnorm, or none")

    @staticmethod
    def weighted_scatter(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
        out = torch.zeros_like(x)
        if edge_index.numel() > 0:
            out.index_add_(0, edge_index[1], x[edge_index[0]] * edge_weight.unsqueeze(1))
        return out

    def _forward_layer(self, x: torch.Tensor, layer: nn.Linear, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if self.model_type == "mlp_table_fallback":
            return layer(x)
        if self.model_type == "weighted_graphsage":
            agg = self.weighted_scatter(x, edge_index, edge_weight)
            return layer(torch.cat([x, agg], dim=1))
        transformed = layer(x)
        if self.model_type == "weighted_sgc":
            return self.weighted_scatter(transformed, edge_index, edge_weight)
        return self.weighted_scatter(transformed, edge_index, edge_weight)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = x.to(torch.float32)
        for idx, layer in enumerate(self.layers):
            prev = h
            h = self._forward_layer(h, layer, edge_index, edge_weight)
            if idx != len(self.layers) - 1:
                h = torch.relu(h)
                h = self.norms[idx](h)
                h = self.dropout(h)
            if self.residual and h.shape == prev.shape:
                h = h + prev
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.embed(x, edge_index, edge_weight)
        return self.classifier(h)
