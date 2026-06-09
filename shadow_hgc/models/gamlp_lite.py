from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class GAMLPResidualGates(nn.Module):
    def __init__(self, block_names: list[str]) -> None:
        super().__init__()
        self.branch_names = [name for name in block_names if name != "self"]
        self.raw_gates = nn.Parameter(torch.zeros(len(self.branch_names)))

    def gate_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.branch_names:
            return torch.empty(0, device=device, dtype=dtype)
        return torch.sigmoid(self.raw_gates.to(device=device, dtype=dtype))

    def gate_values(self) -> dict[str, float]:
        gates = torch.sigmoid(self.raw_gates.detach().cpu())
        return {name: float(gates[idx].item()) for idx, name in enumerate(self.branch_names)}


class GAMLPLiteCore(nn.Module):
    def __init__(self, block_dims: OrderedDict[str, int], *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block_dims = OrderedDict(block_dims)
        if "self" not in self.block_dims:
            raise ValueError("gamlp_lite requires a self block")
        self.block_names = list(self.block_dims)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
                for name, dim in self.block_dims.items()
            }
        )
        self.gates = GAMLPResidualGates(self.block_names)

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        first = blocks["self"]
        h = self.projections["self"](blocks["self"].to(device=first.device, dtype=first.dtype))
        gates = self.gates.gate_tensor(device=first.device, dtype=first.dtype)
        for idx, name in enumerate(self.gates.branch_names):
            branch = self.projections[name](blocks[name].to(device=first.device, dtype=first.dtype))
            gate = gates[idx]
            h = gate * branch + (1.0 - gate) * h
        return h
