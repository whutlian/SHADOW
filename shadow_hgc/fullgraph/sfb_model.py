from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn


class _SFBBlockStandardizer(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
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
            x = x.detach().to(torch.float32)
            self.mean.copy_(x.mean(dim=0))
            self.std.copy_(x.std(dim=0, unbiased=False).clamp_min(self.eps))
            self.fitted = True
            self.source = str(source)
            self.fit_num_rows = int(x.shape[0])

    def freeze(self) -> None:
        if not self.fitted:
            raise RuntimeError("SFB block stats must be fitted before freeze")
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("SFB block stats must be fitted on train target rows before forward")
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype).clamp_min(self.eps)


class BlockGatedResidualTableModel(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        block_norm: Literal["none", "standardize"] = "standardize",
        block_gate: bool = True,
        block_dropout: float = 0.0,
        fusion: Literal["residual_logits", "concat_mlp"] = "residual_logits",
    ) -> None:
        super().__init__()
        if not block_dims:
            raise ValueError("SFB requires at least one block")
        if block_norm not in {"none", "standardize"}:
            raise ValueError("block_norm must be none or standardize")
        if fusion not in {"residual_logits", "concat_mlp"}:
            raise ValueError("fusion must be residual_logits or concat_mlp")
        self.block_dims = OrderedDict((str(key), int(value)) for key, value in block_dims.items())
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.block_norm = block_norm
        self.block_gate = bool(block_gate)
        self.block_dropout = float(block_dropout)
        self.fusion = fusion
        self.block_names = list(self.block_dims)
        self.normalizers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self.logit_heads = nn.ModuleDict()
        for name, dim in self.block_dims.items():
            self.normalizers[name] = _SFBBlockStandardizer(dim) if block_norm == "standardize" else nn.Identity()
            layers: list[nn.Module] = [nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(max(0, int(num_layers) - 1)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            self.mlps[name] = nn.Sequential(*layers)
            self.logit_heads[name] = nn.Linear(hidden_dim, self.num_classes)
        if block_gate:
            self.raw_gates = nn.Parameter(torch.zeros(len(self.block_names)))
        else:
            self.register_buffer("raw_gates", torch.zeros(len(self.block_names)))
        if fusion == "concat_mlp":
            self.concat_head = nn.Sequential(
                nn.Linear(hidden_dim * len(self.block_names), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            self.concat_head = nn.Identity()

    def _check_blocks(self, blocks: dict[str, torch.Tensor]) -> int:
        missing = [name for name in self.block_names if name not in blocks]
        if missing:
            raise ValueError(f"missing SFB blocks: {missing}")
        num_rows = int(blocks[self.block_names[0]].shape[0])
        for name, dim in self.block_dims.items():
            x = blocks[name]
            if x.ndim != 2:
                raise ValueError(f"{name}: SFB block must be rank-2")
            if int(x.shape[0]) != num_rows:
                raise ValueError(f"{name}: SFB row count mismatch")
            if int(x.shape[1]) != dim:
                raise ValueError(f"{name}: expected dim {dim}, got {x.shape[1]}")
        return num_rows

    def _gate_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.block_gate:
            return torch.ones(len(self.block_names), device=device, dtype=dtype)
        return torch.nn.functional.softplus(self.raw_gates.to(device=device, dtype=dtype))

    def fit_block_stats(self, blocks: dict[str, torch.Tensor], *, source: str = "train_target_rows") -> dict:
        self._check_blocks(blocks)
        if source != "train_target_rows":
            raise ValueError("promoted SFB block stats must be fit from train_target_rows")
        for name in self.block_names:
            normalizer = self.normalizers[name]
            if isinstance(normalizer, _SFBBlockStandardizer):
                normalizer.fit(blocks[name], source=source)
        return self.block_norm_metadata()

    def freeze_block_stats(self) -> None:
        for normalizer in self.normalizers.values():
            if isinstance(normalizer, _SFBBlockStandardizer):
                normalizer.freeze()

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check_blocks(blocks)
        first = blocks[self.block_names[0]]
        gates = self._gate_tensor(device=first.device, dtype=first.dtype)
        logits = None
        concat_hidden: list[torch.Tensor] = []
        for idx, name in enumerate(self.block_names):
            x = blocks[name].to(device=first.device, dtype=first.dtype)
            x = self.normalizers[name](x)
            h = self.mlps[name](x)
            if self.training and self.block_dropout > 0.0:
                keep = torch.rand((), device=h.device) > self.block_dropout
                h = h * keep.to(h.dtype)
            if self.fusion == "concat_mlp":
                concat_hidden.append(gates[idx] * h)
            else:
                block_logits = self.logit_heads[name](h)
                logits = gates[idx] * block_logits if logits is None else logits + gates[idx] * block_logits
        if self.fusion == "concat_mlp":
            return self.concat_head(torch.cat(concat_hidden, dim=1))
        return logits

    def block_gate_values(self) -> dict[str, float]:
        gates = self._gate_tensor(device=self.raw_gates.device, dtype=torch.float32).detach().cpu()
        return {name: float(gates[idx].item()) for idx, name in enumerate(self.block_names)}

    def block_norm_metadata(self) -> dict:
        stats = {}
        sources = set()
        fit_rows = 0
        for name in self.block_names:
            normalizer = self.normalizers[name]
            if isinstance(normalizer, _SFBBlockStandardizer):
                stats[name] = {
                    "mean_abs": float(normalizer.mean.abs().mean().item()),
                    "std_mean": float(normalizer.std.mean().item()),
                    "source": normalizer.source,
                    "fitted": bool(normalizer.fitted),
                    "frozen": bool(normalizer.frozen),
                }
                sources.add(normalizer.source)
                fit_rows = max(fit_rows, int(normalizer.fit_num_rows))
        return {
            "block_norm_stats_source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "block_norm_stats": stats,
            "stats_fit_num_rows": int(fit_rows),
        }

    def diagnostics(self) -> dict:
        frozen = True
        for normalizer in self.normalizers.values():
            if isinstance(normalizer, _SFBBlockStandardizer):
                frozen = frozen and normalizer.frozen
        return {
            "model_type": "sfb",
            "sfb_hidden_dim": int(self.hidden_dim),
            "sfb_num_layers": int(self.num_layers),
            "sfb_dropout": float(self.dropout_p),
            "sfb_block_norm": self.block_norm,
            "sfb_block_gate": bool(self.block_gate),
            "sfb_block_dropout": float(self.block_dropout),
            "sfb_fusion": self.fusion,
            "block_dims": dict(self.block_dims),
            "block_gates": self.block_gate_values(),
            "final_logits_activation": "none",
            "stats_frozen": bool(frozen),
            **self.block_norm_metadata(),
        }
