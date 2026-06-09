from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn


class _FeatureBlockStandardizer(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6, lazy_fit: bool = False) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = float(eps)
        self.lazy_fit = bool(lazy_fit)
        self.fitted = False
        self.frozen = False
        self.source = "unfitted"

    def fit(self, x: torch.Tensor, *, source: str) -> None:
        if self.frozen and self.fitted:
            return
        with torch.no_grad():
            x = x.detach().to(torch.float32)
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
                self.fit(x, source="training_forward")
            else:
                raise RuntimeError("block stats must be fitted before forward")
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype).clamp_min(self.eps)


class SeHGNNLite(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        block_norm: str = "standardize",
        block_gate: bool = True,
        fusion: Literal["concat_mlp", "sum_logits"] = "concat_mlp",
        lazy_block_stats: bool = False,
    ) -> None:
        super().__init__()
        if not block_dims:
            raise ValueError("at least one feature block is required")
        if block_norm not in {"none", "standardize"}:
            raise ValueError("block_norm must be none or standardize")
        if fusion not in {"concat_mlp", "sum_logits"}:
            raise ValueError("fusion must be concat_mlp or sum_logits")
        self.block_dims = OrderedDict((str(name), int(dim)) for name, dim in block_dims.items())
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.block_norm = block_norm
        self.block_gate = bool(block_gate)
        self.fusion = fusion
        self.lazy_block_stats = bool(lazy_block_stats)
        self._block_names = list(self.block_dims)
        self.normalizers = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.logit_heads = nn.ModuleDict()
        for idx, (name, dim) in enumerate(self.block_dims.items()):
            key = f"b{idx}"
            self.normalizers[key] = (
                _FeatureBlockStandardizer(dim, lazy_fit=self.lazy_block_stats)
                if block_norm == "standardize"
                else nn.Identity()
            )
            self.projections[key] = nn.Linear(dim, hidden_dim)
            self.logit_heads[key] = nn.Linear(hidden_dim, self.num_classes)
        if self.block_gate:
            self.raw_gates = nn.Parameter(torch.zeros(len(self._block_names)))
        else:
            self.register_buffer("raw_gates", torch.zeros(len(self._block_names)))
        if self.fusion == "concat_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * len(self._block_names), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            self.mlp = nn.Identity()

    def _gate_values_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.block_gate:
            return torch.ones(len(self._block_names), device=device, dtype=dtype)
        return torch.nn.functional.softplus(self.raw_gates.to(device=device, dtype=dtype))

    def _check_blocks(self, blocks: dict[str, torch.Tensor]) -> int:
        missing = [name for name in self._block_names if name not in blocks]
        if missing:
            raise ValueError(f"missing SeHGNN-lite blocks: {missing}")
        num_rows = int(blocks[self._block_names[0]].shape[0])
        for name, dim in self.block_dims.items():
            tensor = blocks[name]
            if tensor.ndim != 2:
                raise ValueError(f"{name}: block must be rank-2")
            if int(tensor.shape[0]) != num_rows:
                raise ValueError(f"{name}: row count mismatch")
            if int(tensor.shape[1]) != dim:
                raise ValueError(f"{name}: expected dim {dim}, got {tensor.shape[1]}")
        return num_rows

    def fit_block_stats(self, blocks: dict[str, torch.Tensor], *, source: str = "train_full_target_rows") -> dict:
        self._check_blocks(blocks)
        for idx, name in enumerate(self._block_names):
            normalizer = self.normalizers[f"b{idx}"]
            if isinstance(normalizer, _FeatureBlockStandardizer):
                normalizer.fit(blocks[name], source=source)
        return self.block_norm_stats()

    def freeze_block_stats(self) -> None:
        for normalizer in self.normalizers.values():
            if isinstance(normalizer, _FeatureBlockStandardizer):
                normalizer.freeze()

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        num_rows = self._check_blocks(blocks)
        first = blocks[self._block_names[0]]
        gates = self._gate_values_tensor(device=first.device, dtype=first.dtype)
        projected: list[torch.Tensor] = []
        logits = None
        for idx, name in enumerate(self._block_names):
            key = f"b{idx}"
            x = blocks[name].to(device=first.device, dtype=first.dtype)
            x = self.normalizers[key](x)
            h = torch.relu(self.projections[key](x))
            h = torch.nn.functional.dropout(h, p=self.dropout_p, training=self.training)
            if self.fusion == "concat_mlp":
                projected.append(gates[idx] * h)
            else:
                block_logits = self.logit_heads[key](h)
                logits = gates[idx] * block_logits if logits is None else logits + gates[idx] * block_logits
        if self.fusion == "sum_logits":
            return logits if logits is not None else torch.empty(num_rows, self.num_classes, device=first.device)
        return self.mlp(torch.cat(projected, dim=1))

    def block_gate_values(self) -> dict[str, float]:
        gates = self._gate_values_tensor(device=self.raw_gates.device, dtype=torch.float32).detach().cpu()
        return {name: float(gates[idx].item()) for idx, name in enumerate(self._block_names)}

    def block_norm_stats(self) -> dict[str, dict]:
        stats = {}
        for idx, name in enumerate(self._block_names):
            normalizer = self.normalizers[f"b{idx}"]
            if isinstance(normalizer, _FeatureBlockStandardizer):
                stats[name] = {
                    "mean_abs": float(normalizer.mean.abs().mean().item()),
                    "std_mean": float(normalizer.std.mean().item()),
                    "source": normalizer.source,
                    "fitted": bool(normalizer.fitted),
                    "frozen": bool(normalizer.frozen),
                }
        return stats

    def diagnostics(self) -> dict:
        sources = {value["source"] for value in self.block_norm_stats().values()}
        return {
            "model_type": "sehgnn_lite",
            "block_dims": dict(self.block_dims),
            "block_gates": self.block_gate_values(),
            "block_norm_stats": self.block_norm_stats(),
            "block_norm_stats_source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "fusion": self.fusion,
            "final_logits_activation": "none",
        }
