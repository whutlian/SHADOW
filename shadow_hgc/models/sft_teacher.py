from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn

from shadow_hgc.features.block_stats import BlockStandardizer
from shadow_hgc.models.gamlp_lite import GAMLPLiteCore
from shadow_hgc.models.hop_attention import BlockAttention


def _contains_forbidden_logit_name(name: str) -> bool:
    lowered = name.lower()
    return "logit" in lowered or "kd" in lowered or "teacher_logits" in lowered


class _TorchBlockStandardizer(nn.Module):
    def __init__(self, dim: int, *, name: str) -> None:
        super().__init__()
        self.name = name
        self.register_buffer("mean", torch.zeros(int(dim)))
        self.register_buffer("std", torch.ones(int(dim)))
        self.fitted = False
        self.frozen = False
        self.fit_scope = "unfitted"
        self.fit_rows: list[int] = []

    def fit(self, block: torch.Tensor, *, train_rows: torch.Tensor) -> None:
        stats = BlockStandardizer.fit(block, train_rows=train_rows, block_name=self.name).freeze()
        self.mean.copy_(stats.mean.to(self.mean.device, self.mean.dtype))
        self.std.copy_(stats.std.to(self.std.device, self.std.dtype))
        self.fit_scope = stats.fit_scope
        self.fit_rows = stats.fit_rows
        self.fitted = True
        self.frozen = True

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("SFT block stats must be fitted before forward")
        return torch.nan_to_num(
            (block - self.mean.to(block.device, block.dtype)) / self.std.to(block.device, block.dtype).clamp_min(1e-6),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def metadata(self) -> dict:
        if len(self.fit_rows) <= 1024:
            row_payload = {"fit_rows": self.fit_rows}
        else:
            row_payload = {
                "fit_rows": [],
                "fit_rows_count": len(self.fit_rows),
                "fit_rows_head": self.fit_rows[:16],
                "fit_rows_tail": self.fit_rows[-16:],
            }
        return {
            "source": self.fit_scope,
            "fitted": bool(self.fitted),
            "frozen": bool(self.frozen),
            "mean_abs": float(self.mean.abs().mean().item()) if self.mean.numel() else 0.0,
            "std_mean": float(self.std.mean().item()) if self.std.numel() else 0.0,
            **row_payload,
        }


class SAGNLiteCore(nn.Module):
    def __init__(self, block_dims: OrderedDict[str, int], *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block_dims = OrderedDict(block_dims)
        self.block_names = list(self.block_dims)
        self.mlps = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
                for name, dim in self.block_dims.items()
            }
        )
        self.attention = BlockAttention(self.block_names, int(hidden_dim))

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        first = blocks[self.block_names[0]]
        hidden = {name: self.mlps[name](blocks[name].to(device=first.device, dtype=first.dtype)) for name in self.block_names}
        return self.attention(hidden)

    def gate_values(self) -> dict[str, float]:
        return self.attention.gate_values()

    def attention_mean(self) -> dict[str, float]:
        return dict(self.attention.last_attention_mean)


class SFTTableTeacher(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        model_type: Literal["sagn_lite", "gamlp_lite"] = "sagn_lite",
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
        if model_type not in {"sagn_lite", "gamlp_lite"}:
            raise ValueError("model_type must be sagn_lite or gamlp_lite")
        self.model_type = model_type
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.normalizers = nn.ModuleDict({name: _TorchBlockStandardizer(dim, name=name) for name, dim in self.block_dims.items()})
        if model_type == "sagn_lite":
            self.core = SAGNLiteCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.core = GAMLPLiteCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
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
        if self.model_type == "sagn_lite":
            return self.core.gate_values()
        return self.core.gates.gate_values()

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
        if self.model_type == "sagn_lite":
            diag["attention_mean"] = self.core.attention_mean()
        return diag
