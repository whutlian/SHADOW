from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from shadow_hgc.features.compiled_table import CompiledDemandSchema, block_slices, schema_to_dict


class _BlockStandardizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps
        self.fitted = False

    def fit(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            self.mean.copy_(x.mean(dim=0))
            self.std.copy_(x.std(dim=0, unbiased=False).clamp_min(self.eps))
            self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted and self.training and x.numel() > 0:
            self.fit(x.detach())
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype).clamp_min(self.eps)


class CompiledDemandMLP(nn.Module):
    def __init__(
        self,
        schema: CompiledDemandSchema,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        block_norm: str = "standardize",
        block_gate: bool = True,
        fusion: Literal["concat_mlp", "sum_logits"] = "concat_mlp",
    ) -> None:
        super().__init__()
        if fusion not in {"concat_mlp", "sum_logits"}:
            raise ValueError("fusion must be concat_mlp or sum_logits")
        if block_norm not in {"none", "standardize"}:
            raise ValueError("block_norm must be none or standardize")
        self.schema = schema
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.block_norm = block_norm
        self.block_gate = bool(block_gate)
        self.fusion = fusion
        self._slices = block_slices(schema)
        self.normalizers = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.logit_heads = nn.ModuleDict()
        gate_names = []
        for idx, block in enumerate(schema.blocks):
            key = f"b{idx}"
            gate_names.append(block.name)
            self.normalizers[key] = _BlockStandardizer(block.dim) if block_norm == "standardize" else nn.Identity()
            self.projections[key] = nn.Linear(block.dim, hidden_dim)
            self.logit_heads[key] = nn.Linear(hidden_dim, num_classes)
        self._gate_names = gate_names
        if block_gate:
            self.raw_gates = nn.Parameter(torch.zeros(len(schema.blocks)))
        else:
            self.register_buffer("raw_gates", torch.zeros(len(schema.blocks)))
        if fusion == "concat_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * len(schema.blocks), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.mlp = nn.Identity()

    def _gate_values_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.block_gate:
            return torch.ones(len(self.schema.blocks), device=device, dtype=dtype)
        return torch.nn.functional.softplus(self.raw_gates.to(device=device, dtype=dtype))

    def forward(self, table: torch.Tensor) -> torch.Tensor:
        if table.shape[1] != self.schema.total_dim:
            raise ValueError(f"expected compiled dim {self.schema.total_dim}, got {table.shape[1]}")
        projected = []
        logits = None
        gates = self._gate_values_tensor(device=table.device, dtype=table.dtype)
        for idx, block in enumerate(self.schema.blocks):
            key = f"b{idx}"
            block_x = table[:, self._slices[block.name]]
            block_x = self.normalizers[key](block_x)
            h = torch.relu(self.projections[key](block_x))
            h = torch.nn.functional.dropout(h, p=self.dropout_p, training=self.training)
            gated_h = gates[idx] * h
            if self.fusion == "concat_mlp":
                projected.append(gated_h)
            else:
                block_logits = self.logit_heads[key](h)
                logits = gates[idx] * block_logits if logits is None else logits + gates[idx] * block_logits
        if self.fusion == "sum_logits":
            return logits if logits is not None else torch.empty(table.shape[0], self.num_classes, device=table.device)
        return self.mlp(torch.cat(projected, dim=1))

    def block_gate_values(self) -> dict[str, float]:
        gates = self._gate_values_tensor(device=self.raw_gates.device, dtype=torch.float32).detach().cpu()
        return {name: float(gates[idx].item()) for idx, name in enumerate(self._gate_names)}

    def block_norm_stats(self) -> dict[str, dict]:
        stats = {}
        for idx, block in enumerate(self.schema.blocks):
            normalizer = self.normalizers[f"b{idx}"]
            if isinstance(normalizer, _BlockStandardizer):
                stats[block.name] = {
                    "mean_abs": float(normalizer.mean.abs().mean().item()),
                    "std_mean": float(normalizer.std.mean().item()),
                }
        return stats

    def diagnostics(self) -> dict:
        return {
            "compiled_head": True,
            "compiled_schema_total_dim": int(self.schema.total_dim),
            "compiled_blocks": schema_to_dict(self.schema)["blocks"],
            "final_logits_activation": "none",
            "block_gates": self.block_gate_values(),
            "block_norm_stats": self.block_norm_stats(),
            "fusion": self.fusion,
        }
