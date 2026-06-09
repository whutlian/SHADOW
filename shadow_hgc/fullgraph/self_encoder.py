from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be relu or gelu")


def _norm(name: str, dim: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError("norm must be none, batchnorm, or layernorm")


@dataclass
class SelfEncoderOutput:
    self_logits: torch.Tensor
    self_hidden: torch.Tensor


class StrongSelfEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        norm: str = "layernorm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if int(num_layers) < 2:
            raise ValueError("StrongSelfEncoder requires at least 2 layers")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers) - 1):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), _norm(norm, int(hidden_dim)), _activation(activation), nn.Dropout(float(dropout))])
            in_dim = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> SelfEncoderOutput:
        hidden = self.encoder(x.to(torch.float32))
        return SelfEncoderOutput(self_logits=self.head(hidden), self_hidden=hidden)
