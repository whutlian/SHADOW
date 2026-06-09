from __future__ import annotations

import torch
from torch import nn


class BlockAttention(nn.Module):
    def __init__(self, block_names: list[str], hidden_dim: int) -> None:
        super().__init__()
        self.block_names = list(block_names)
        self.scorers = nn.ModuleDict({name: nn.Linear(int(hidden_dim), 1) for name in self.block_names})
        self.raw_block_bias = nn.Parameter(torch.zeros(len(self.block_names)))
        self.last_attention_mean: dict[str, float] = {name: 0.0 for name in self.block_names}

    def forward(self, hidden_by_block: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.block_names:
            raise ValueError("at least one block is required")
        scores = []
        hiddens = []
        first = hidden_by_block[self.block_names[0]]
        for idx, name in enumerate(self.block_names):
            h = hidden_by_block[name].to(device=first.device, dtype=first.dtype)
            hiddens.append(h)
            scores.append(self.scorers[name](h) + self.raw_block_bias[idx].to(device=h.device, dtype=h.dtype))
        alpha = torch.softmax(torch.cat(scores, dim=1), dim=1)
        stacked = torch.stack(hiddens, dim=1)
        self.last_attention_mean = {
            name: float(alpha[:, idx].detach().mean().cpu().item()) if alpha.numel() else 0.0
            for idx, name in enumerate(self.block_names)
        }
        return (alpha.unsqueeze(2) * stacked).sum(dim=1)

    def gate_values(self) -> dict[str, float]:
        gates = torch.softmax(self.raw_block_bias.detach().cpu(), dim=0)
        return {name: float(gates[idx].item()) for idx, name in enumerate(self.block_names)}
