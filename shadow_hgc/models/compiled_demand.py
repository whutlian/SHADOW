from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from shadow_hgc.features.compiled_table import CompiledDemandSchema, block_slices, schema_to_dict


class _BlockStandardizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, lazy_fit: bool = True) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps
        self.lazy_fit = bool(lazy_fit)
        self.fitted = False
        self.frozen = False
        self.source = "unfitted"

    def fit(self, x: torch.Tensor, *, source: str = "training_forward") -> None:
        if self.frozen and self.fitted:
            return
        with torch.no_grad():
            self.mean.copy_(x.mean(dim=0))
            self.std.copy_(x.std(dim=0, unbiased=False).clamp_min(self.eps))
            self.fitted = True
            self.source = source

    def freeze(self) -> None:
        if not self.fitted:
            raise RuntimeError("block stats must be fitted before freezing")
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            if self.training and self.lazy_fit and x.numel() > 0:
                self.fit(x.detach(), source="training_forward")
            else:
                raise RuntimeError("block stats must be fitted before forward when lazy fitting is disabled")
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
        lazy_block_stats: bool = True,
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
        self.lazy_block_stats = bool(lazy_block_stats)
        self._slices = block_slices(schema)
        self.normalizers = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.logit_heads = nn.ModuleDict()
        gate_names = []
        for idx, block in enumerate(schema.blocks):
            key = f"b{idx}"
            gate_names.append(block.name)
            self.normalizers[key] = (
                _BlockStandardizer(block.dim, lazy_fit=self.lazy_block_stats)
                if block_norm == "standardize"
                else nn.Identity()
            )
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

    def fit_block_stats(self, table: torch.Tensor, *, source: str = "train_full_demand_table") -> dict:
        if table.shape[1] != self.schema.total_dim:
            raise ValueError(f"expected compiled dim {self.schema.total_dim}, got {table.shape[1]}")
        for idx, block in enumerate(self.schema.blocks):
            normalizer = self.normalizers[f"b{idx}"]
            if isinstance(normalizer, _BlockStandardizer):
                normalizer.fit(table[:, self._slices[block.name]].detach().to(torch.float32), source=source)
        return self.block_norm_stats()

    def freeze_block_stats(self) -> None:
        for normalizer in self.normalizers.values():
            if isinstance(normalizer, _BlockStandardizer):
                normalizer.freeze()

    def block_norm_stats(self) -> dict[str, dict]:
        stats = {}
        for idx, block in enumerate(self.schema.blocks):
            normalizer = self.normalizers[f"b{idx}"]
            if isinstance(normalizer, _BlockStandardizer):
                stats[block.name] = {
                    "mean": normalizer.mean.detach().cpu().tolist(),
                    "std": normalizer.std.detach().cpu().tolist(),
                    "mean_abs": float(normalizer.mean.abs().mean().item()),
                    "std_mean": float(normalizer.std.mean().item()),
                    "fitted": bool(normalizer.fitted),
                    "frozen": bool(normalizer.frozen),
                    "source": normalizer.source,
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
            "lazy_block_stats": bool(self.lazy_block_stats),
        }


def fit_compiled_block_stats(
    model: CompiledDemandMLP,
    train_full_table: torch.Tensor,
    schema: CompiledDemandSchema,
) -> dict:
    """Fit and freeze per-block stats from original train-target demand rows."""

    if model.schema != schema:
        raise ValueError("compiled block stats schema must match model schema")
    model.fit_block_stats(train_full_table, source="train_full_demand_table")
    model.freeze_block_stats()
    return {
        "compiled_block_stats_source": "train_full_demand_table",
        "block_norm_stats": model.block_norm_stats(),
    }


def fit_block_stats(
    train_full_table: torch.Tensor,
    schema: CompiledDemandSchema,
    *,
    source: str = "train_full_target_demand_table",
    eps: float = 1e-6,
) -> dict:
    slices = block_slices(schema)
    blocks = {}
    for block in schema.blocks:
        x = train_full_table[:, slices[block.name]].detach().to(torch.float32)
        blocks[block.name] = {
            "mean": x.mean(dim=0),
            "std": x.std(dim=0, unbiased=False).clamp_min(float(eps)),
            "dim": int(block.dim),
        }
    return {
        "source": source,
        "schema_total_dim": int(schema.total_dim),
        "fit_num_rows": int(train_full_table.shape[0]),
        "blocks": blocks,
        "frozen": False,
    }


def freeze_block_stats(stats: dict) -> dict:
    frozen = dict(stats)
    frozen["frozen"] = True
    return frozen


def _stats_metadata(schema: CompiledDemandSchema, stats: dict) -> dict:
    block_names = [block.name for block in schema.blocks]
    std_values = []
    mean_norms = {}
    for block in schema.blocks:
        entry = stats["blocks"][block.name]
        mean = entry["mean"]
        std = entry["std"]
        mean_norms[block.name] = float(mean.norm().item())
        std_values.append(std.flatten())
    all_std = torch.cat(std_values) if std_values else torch.ones(1)
    return {
        "block_norm_stats_source": stats.get("source", ""),
        "block_names": block_names,
        "block_dims": {block.name: int(block.dim) for block in schema.blocks},
        "block_mean_norms": mean_norms,
        "block_std_min": float(all_std.min().item()),
        "block_std_max": float(all_std.max().item()),
        "stats_fit_num_rows": int(stats.get("fit_num_rows", 0)),
        "stats_frozen": bool(stats.get("frozen", False)),
    }


def apply_block_stats(
    table: torch.Tensor,
    schema: CompiledDemandSchema,
    stats: dict,
) -> tuple[torch.Tensor, dict]:
    if int(stats.get("schema_total_dim", schema.total_dim)) != int(schema.total_dim):
        raise ValueError("compiled block stats schema dimension mismatch")
    slices = block_slices(schema)
    out = table.clone().to(torch.float32)
    for block in schema.blocks:
        entry = stats["blocks"][block.name]
        mean = entry["mean"].to(device=out.device, dtype=out.dtype)
        std = entry["std"].to(device=out.device, dtype=out.dtype).clamp_min(1e-12)
        out[:, slices[block.name]] = (out[:, slices[block.name]] - mean) / std
    return out, _stats_metadata(schema, stats)
