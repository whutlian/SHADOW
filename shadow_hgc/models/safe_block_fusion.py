from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn


class _Standardizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(dim)))
        self.register_buffer("std", torch.ones(int(dim)))
        self.eps = float(eps)
        self.fitted = False
        self.source = "unfitted"

    def fit(self, x: torch.Tensor, *, source: str) -> None:
        with torch.no_grad():
            value = x.detach().to(torch.float32)
            self.mean.copy_(value.mean(dim=0))
            self.std.copy_(value.std(dim=0, unbiased=False).clamp_min(self.eps))
            self.fitted = True
            self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return x
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype).clamp_min(self.eps)


def _head(dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_dim), int(num_classes)),
    )


class SafeBlockFusionClassifier(nn.Module):
    """Residual additive block classifier with non-self gates initialized off."""

    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        block_norm: Literal["standardize", "none"] = "standardize",
        non_self_gate_init: float = -8.0,
    ) -> None:
        super().__init__()
        if "self" not in block_dims:
            raise ValueError("SafeBlockFusionClassifier requires a self block")
        self.block_dims = OrderedDict((str(k), int(v)) for k, v in block_dims.items())
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.block_norm = block_norm
        self.self_head = _head(self.block_dims["self"], hidden_dim, num_classes, dropout)
        self.branch_names = [name for name in self.block_dims if name != "self"]
        self.branch_heads = nn.ModuleDict({name: _head(self.block_dims[name], hidden_dim, num_classes, dropout) for name in self.branch_names})
        self.normalizers = nn.ModuleDict({
            name: _Standardizer(dim) if block_norm == "standardize" else nn.Identity()
            for name, dim in self.block_dims.items()
        })
        self.raw_gates = nn.Parameter(torch.full((len(self.branch_names),), float(non_self_gate_init)))

    def _check(self, blocks: dict[str, torch.Tensor]) -> None:
        for name, dim in self.block_dims.items():
            if name not in blocks:
                raise ValueError(f"missing block {name}")
            if blocks[name].ndim != 2 or int(blocks[name].shape[1]) != dim:
                raise ValueError(f"block {name} expected dim {dim}, got {tuple(blocks[name].shape)}")

    def fit_block_stats(self, blocks: dict[str, torch.Tensor], *, source: str = "train_target_rows") -> dict:
        self._check(blocks)
        for name in self.block_dims:
            normalizer = self.normalizers[name]
            if isinstance(normalizer, _Standardizer):
                normalizer.fit(blocks[name], source=source)
        return self.block_norm_metadata()

    def gate_values(self) -> dict[str, float]:
        gates = torch.nn.functional.softplus(self.raw_gates.detach().cpu())
        return {name: float(gates[i].item()) for i, name in enumerate(self.branch_names)}

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check(blocks)
        first = blocks["self"]
        self_x = self.normalizers["self"](first.to(torch.float32))
        logits = self.self_head(self_x)
        gates = torch.nn.functional.softplus(self.raw_gates.to(logits.device, logits.dtype))
        for index, name in enumerate(self.branch_names):
            x = blocks[name].to(device=logits.device, dtype=logits.dtype)
            x = self.normalizers[name](x)
            logits = logits + gates[index] * self.branch_heads[name](x)
        return logits

    def block_norm_metadata(self) -> dict:
        stats = {}
        sources = set()
        for name, normalizer in self.normalizers.items():
            if isinstance(normalizer, _Standardizer):
                sources.add(normalizer.source)
                stats[name] = {
                    "source": normalizer.source,
                    "fitted": bool(normalizer.fitted),
                    "mean_abs": float(normalizer.mean.abs().mean().item()),
                    "std_mean": float(normalizer.std.mean().item()),
                }
        return {
            "block_norm_stats_source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "block_norm_stats": stats,
        }

    def diagnostics(self) -> dict:
        return {
            "model_type": "safe_block_fusion",
            "block_dims": dict(self.block_dims),
            "block_gates": self.gate_values(),
            "block_norm": self.block_norm,
            "final_logits_activation": "none",
            **self.block_norm_metadata(),
        }
