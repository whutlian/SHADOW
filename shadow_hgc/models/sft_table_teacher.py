from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn

from shadow_hgc.models.gamlp_lite import GAMLPLiteCore
from shadow_hgc.models.sft_teacher import SAGNLiteCore, _TorchBlockStandardizer, _contains_forbidden_logit_name


class ResidualBlockGatedCore(nn.Module):
    def __init__(self, block_dims: OrderedDict[str, int], *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block_dims = OrderedDict(block_dims)
        self.block_names = list(self.block_dims)
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
                for name, dim in self.block_dims.items()
            }
        )
        self.raw_gates = nn.Parameter(torch.zeros(len(self.block_names)))
        self.residual = nn.Linear(self.block_dims["self"], int(hidden_dim), bias=False) if "self" in self.block_dims else None

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        first = blocks[self.block_names[0]]
        gates = torch.softmax(self.raw_gates.to(device=first.device, dtype=first.dtype), dim=0)
        hidden = None
        for idx, name in enumerate(self.block_names):
            branch = self.branches[name](blocks[name].to(device=first.device, dtype=first.dtype))
            hidden = gates[idx] * branch if hidden is None else hidden + gates[idx] * branch
        if self.residual is not None:
            hidden = hidden + self.residual(blocks["self"].to(device=first.device, dtype=first.dtype))
        return hidden

    def gate_values(self) -> dict[str, float]:
        gates = torch.softmax(self.raw_gates.detach().cpu(), dim=0)
        return {name: float(gates[idx].item()) for idx, name in enumerate(self.block_names)}


class SFTTableTeacherV2(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        model_type: Literal["sagn_lite", "gamlp_lite", "residual_block_gated"] = "sagn_lite",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers_per_block: int = 1,
    ) -> None:
        super().__init__()
        del num_layers_per_block
        if not block_dims:
            raise ValueError("at least one SFT block is required")
        forbidden = [name for name in block_dims if _contains_forbidden_logit_name(name)]
        if forbidden:
            raise ValueError(f"logits are forbidden as T2-SFT-NL input blocks: {forbidden}")
        self.block_dims = OrderedDict((str(name), int(dim)) for name, dim in block_dims.items())
        if model_type == "gamlp_lite" and "self" not in self.block_dims:
            raise ValueError("gamlp_lite requires a self block")
        if model_type not in {"sagn_lite", "gamlp_lite", "residual_block_gated"}:
            raise ValueError("model_type must be sagn_lite, gamlp_lite, or residual_block_gated")
        self.model_type = str(model_type)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.normalizers = nn.ModuleDict({name: _TorchBlockStandardizer(dim, name=name) for name, dim in self.block_dims.items()})
        if model_type == "sagn_lite":
            self.core = SAGNLiteCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        elif model_type == "gamlp_lite":
            self.core = GAMLPLiteCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.core = ResidualBlockGatedCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))

    def _check(self, blocks: dict[str, torch.Tensor]) -> None:
        for name, dim in self.block_dims.items():
            if name not in blocks:
                raise ValueError(f"missing SFT block {name}")
            if blocks[name].ndim != 2 or int(blocks[name].shape[1]) != int(dim):
                raise ValueError(f"{name}: expected dim {dim}, got {tuple(blocks[name].shape)}")

    def fit_block_stats(self, blocks: dict[str, torch.Tensor], *, train_rows: torch.Tensor | list[int]) -> dict:
        self._check(blocks)
        rows = torch.as_tensor(train_rows, dtype=torch.long)
        for name in self.block_dims:
            self.normalizers[name].fit(blocks[name].detach().to(torch.float32), train_rows=rows)
        return self.block_norm_metadata()

    def _normalized(self, blocks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.normalizers[name](blocks[name].to(torch.float32)) for name in self.block_dims}

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check(blocks)
        hidden = self.core(self._normalized(blocks))
        return self.classifier(hidden)

    def gate_values(self) -> dict[str, float]:
        if hasattr(self.core, "gate_values"):
            return self.core.gate_values()
        return {}

    def block_norm_metadata(self) -> dict:
        return {name: self.normalizers[name].metadata() for name in self.block_dims}

    def diagnostics(self) -> dict:
        diag = {
            "model_type": self.model_type,
            "block_dims": dict(self.block_dims),
            "block_gates": self.gate_values(),
            "block_norm_stats": self.block_norm_metadata(),
            "block_norm_stats_source": "train_target_rows",
            "uses_logits_as_input": False,
            "uses_teacher_logits": False,
            "uses_kd": False,
            "uses_full_graph_backprop": False,
            "final_logits_activation": "none",
        }
        if self.model_type == "sagn_lite" and hasattr(self.core, "attention_mean"):
            diag["attention_mean"] = self.core.attention_mean()
        return diag


SFTTableTeacher = SFTTableTeacherV2
