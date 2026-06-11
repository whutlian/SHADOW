from __future__ import annotations

import torch
from torch import nn

from shadow_hgc.reddit.graph_student import WeightedGraphStudent


class WeightedOperatorStudent(WeightedGraphStudent):
    """T29 operator student using explicit pre-normalized edge weights."""

    uses_library_normalization = False


class OperatorSFTTableHead(nn.Module):
    """Small table head for OMCP-derived SFT blocks."""

    def __init__(self, *, input_dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(int(input_dim), int(num_classes))
        else:
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(num_classes)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))
