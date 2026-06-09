from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn


class _BlockStandardizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(dim)))
        self.register_buffer("std", torch.ones(int(dim)))
        self.eps = float(eps)
        self.fitted = False
        self.frozen = False
        self.source = "unfitted"
        self.fit_num_rows = 0

    def fit(self, x: torch.Tensor, *, source: str) -> None:
        if self.frozen and self.fitted:
            return
        with torch.no_grad():
            scoped = x.detach().to(torch.float32)
            self.mean.copy_(scoped.mean(dim=0))
            self.std.copy_(scoped.std(dim=0, unbiased=False).clamp_min(self.eps))
            self.fitted = True
            self.source = source
            self.fit_num_rows = int(scoped.shape[0])

    def freeze(self) -> None:
        if not self.fitted:
            raise RuntimeError("block stats must be fitted before freeze")
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("block stats must be fitted before forward")
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype).clamp_min(self.eps)


def _make_mlp(dim: int, hidden_dim: int, num_classes: int, dropout: float, norm: str, activation: str) -> nn.Sequential:
    if activation == "relu":
        act: nn.Module = nn.ReLU()
    elif activation == "gelu":
        act = nn.GELU()
    else:
        raise ValueError("activation must be relu or gelu")
    layers: list[nn.Module] = [nn.Linear(int(dim), int(hidden_dim))]
    if norm == "layernorm":
        layers.append(nn.LayerNorm(int(hidden_dim)))
    elif norm == "batchnorm":
        layers.append(nn.BatchNorm1d(int(hidden_dim)))
    elif norm != "none":
        raise ValueError("norm must be none, layernorm, or batchnorm")
    layers.extend([act, nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(num_classes))])
    return nn.Sequential(*layers)


class BlockGatedTableModel(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        hidden_dim: int = 256,
        branch_hidden_dim: int | None = None,
        branch_dropout: float = 0.3,
        block_norm: Literal["standardize", "none"] = "standardize",
        branch_norm: str = "none",
        activation: str = "relu",
        block_gates: bool = True,
        fusion_type: Literal["residual_sum", "concat_mlp"] = "residual_sum",
    ) -> None:
        super().__init__()
        if "self" not in block_dims:
            raise ValueError("SFB-v2 requires a self block")
        self.block_dims = OrderedDict((str(k), int(v)) for k, v in block_dims.items())
        self.block_names = list(self.block_dims)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.branch_hidden_dim = int(branch_hidden_dim or hidden_dim)
        self.block_norm = block_norm
        self.branch_dropout = float(branch_dropout)
        self.block_gates_enabled = bool(block_gates)
        self.fusion_type = fusion_type
        self.normalizers = nn.ModuleDict()
        self.branches = nn.ModuleDict()
        for name, dim in self.block_dims.items():
            self.normalizers[name] = _BlockStandardizer(dim) if block_norm == "standardize" else nn.Identity()
            self.branches[name] = _make_mlp(dim, self.branch_hidden_dim, self.num_classes, branch_dropout, branch_norm, activation)
        branch_names = [name for name in self.block_names if name != "self"]
        if block_gates:
            self.raw_gates = nn.Parameter(torch.zeros(len(branch_names)))
        else:
            self.register_buffer("raw_gates", torch.zeros(len(branch_names)))
        self.branch_gate_names = branch_names

    def _check(self, blocks: dict[str, torch.Tensor]) -> None:
        for name, dim in self.block_dims.items():
            if name not in blocks:
                raise ValueError(f"missing block {name}")
            if blocks[name].ndim != 2 or int(blocks[name].shape[1]) != dim:
                raise ValueError(f"block {name} expected dim {dim}, got {tuple(blocks[name].shape)}")

    def fit_block_stats(self, blocks: dict[str, torch.Tensor], *, source: str = "train_target_rows") -> dict:
        if source != "train_target_rows":
            raise ValueError("SFB-v2 block stats must be fit on train_target_rows")
        self._check(blocks)
        for name in self.block_names:
            norm = self.normalizers[name]
            if isinstance(norm, _BlockStandardizer):
                norm.fit(blocks[name], source=source)
        return self.block_norm_metadata()

    def freeze_block_stats(self) -> None:
        for norm in self.normalizers.values():
            if isinstance(norm, _BlockStandardizer):
                norm.freeze()

    def gate_values(self) -> dict[str, float]:
        if not self.branch_gate_names:
            return {}
        if not self.block_gates_enabled:
            return {name: 1.0 for name in self.branch_gate_names}
        gates = torch.nn.functional.softplus(self.raw_gates.detach().cpu())
        return {name: float(gates[i].item()) for i, name in enumerate(self.branch_gate_names)}

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check(blocks)
        first = blocks["self"]
        logits = None
        if self.block_gates_enabled and self.branch_gate_names:
            gate_tensor = torch.nn.functional.softplus(self.raw_gates.to(device=first.device, dtype=first.dtype))
        else:
            gate_tensor = torch.ones(len(self.branch_gate_names), device=first.device, dtype=first.dtype)
        gate_lookup = {name: gate_tensor[idx] for idx, name in enumerate(self.branch_gate_names)}
        for name in self.block_names:
            x = blocks[name].to(device=first.device, dtype=first.dtype)
            x = self.normalizers[name](x)
            branch_logits = self.branches[name](x)
            if name == "self":
                logits = branch_logits
            else:
                gate = gate_lookup[name]
                logits = logits + gate * branch_logits
        return logits

    def block_norm_metadata(self) -> dict:
        sources = set()
        fit_rows = 0
        stats = {}
        for name in self.block_names:
            norm = self.normalizers[name]
            if isinstance(norm, _BlockStandardizer):
                sources.add(norm.source)
                fit_rows = max(fit_rows, int(norm.fit_num_rows))
                stats[name] = {
                    "source": norm.source,
                    "fitted": bool(norm.fitted),
                    "frozen": bool(norm.frozen),
                    "mean_abs": float(norm.mean.abs().mean().item()),
                    "std_mean": float(norm.std.mean().item()),
                }
        return {
            "block_norm_stats_source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "block_norm_stats": stats,
            "stats_fit_num_rows": int(fit_rows),
        }

    def diagnostics(self) -> dict:
        frozen = True
        for norm in self.normalizers.values():
            if isinstance(norm, _BlockStandardizer):
                frozen = frozen and norm.frozen
        return {
            "model_type": "sfb_v2",
            "block_dims": dict(self.block_dims),
            "block_gates": self.gate_values(),
            "block_norm": self.block_norm,
            "fusion_type": self.fusion_type,
            "final_logits_activation": "none",
            "stats_frozen": bool(frozen),
            **self.block_norm_metadata(),
        }
