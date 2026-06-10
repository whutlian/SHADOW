from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from shadow_hgc.models.hop_attention import BlockAttention


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.3,
    activation: str = "relu",
    norm: str = "none",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    act: nn.Module
    for _ in range(max(1, int(num_layers))):
        layers.append(nn.Linear(current, int(hidden_dim)))
        if norm == "batchnorm":
            layers.append(nn.BatchNorm1d(int(hidden_dim)))
        elif norm == "layernorm":
            layers.append(nn.LayerNorm(int(hidden_dim)))
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        layers.append(act)
        layers.append(nn.Dropout(float(dropout)))
        current = int(hidden_dim)
    return nn.Sequential(*layers)


class SAGNLiteV2Core(nn.Module):
    def __init__(
        self,
        block_dims: OrderedDict[str, int],
        *,
        hidden_dim: int,
        dropout: float,
        block_dropout: float = 0.0,
        hop_dropout: float = 0.0,
        num_layers: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.block_dims = OrderedDict(block_dims)
        self.block_names = list(self.block_dims)
        self.block_dropout = float(block_dropout)
        self.hop_dropout = nn.Dropout(float(hop_dropout))
        self.encoders = nn.ModuleDict(
            {
                name: make_mlp(dim, hidden_dim, num_layers=num_layers, dropout=dropout, activation=activation, norm=norm)
                for name, dim in self.block_dims.items()
            }
        )
        self.attention = BlockAttention(self.block_names, int(hidden_dim))

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        first = blocks[self.block_names[0]]
        hidden: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            branch = self.encoders[name](blocks[name].to(device=first.device, dtype=first.dtype))
            if self.training and self.block_dropout > 0.0 and name != "self":
                keep = (torch.rand(branch.shape[0], 1, device=branch.device) >= self.block_dropout).to(branch.dtype)
                branch = branch * keep / max(1e-12, 1.0 - self.block_dropout)
            hidden[name] = self.hop_dropout(branch)
        return self.attention(hidden)

    def gate_values(self) -> dict[str, float]:
        return self.attention.gate_values()

    def attention_mean(self) -> dict[str, float]:
        return dict(self.attention.last_attention_mean)
