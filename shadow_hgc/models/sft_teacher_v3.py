from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn

from shadow_hgc.models.gamlp_lite import GAMLPLiteCore
from shadow_hgc.models.sagn_lite import SAGNLiteV2Core, make_mlp
from shadow_hgc.models.sft_teacher import _TorchBlockStandardizer, _contains_forbidden_logit_name
from shadow_hgc.models.sft_table_teacher import ResidualBlockGatedCore


class GAMLPRecursiveV2Core(nn.Module):
    def __init__(
        self,
        block_dims: OrderedDict[str, int],
        *,
        hidden_dim: int,
        dropout: float,
        num_layers: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ) -> None:
        super().__init__()
        if "self" not in block_dims:
            raise ValueError("gamlp_recursive_v2 requires a self block")
        self.block_dims = OrderedDict(block_dims)
        self.branch_names = [name for name in self.block_dims if name != "self"]
        self.projections = nn.ModuleDict(
            {
                name: make_mlp(dim, hidden_dim, num_layers=num_layers, dropout=dropout, activation=activation, norm=norm)
                for name, dim in self.block_dims.items()
            }
        )
        self.raw_gates = nn.Parameter(torch.zeros(len(self.branch_names)))
        self.self_residual = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        first = blocks["self"]
        h = self.projections["self"](blocks["self"].to(device=first.device, dtype=first.dtype))
        gates = torch.sigmoid(self.raw_gates.to(device=h.device, dtype=h.dtype))
        for idx, name in enumerate(self.branch_names):
            branch = self.projections[name](blocks[name].to(device=h.device, dtype=h.dtype))
            h = self.self_residual(h) + gates[idx] * branch + (1.0 - gates[idx]) * h
        return h

    def gate_values(self) -> dict[str, float]:
        gates = torch.sigmoid(self.raw_gates.detach().cpu())
        return {name: float(gates[idx].item()) for idx, name in enumerate(self.branch_names)}


class SFTTeacherV3(nn.Module):
    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        model_type: Literal[
            "sagn_lite_v2",
            "gamlp_lite_v2",
            "gamlp_recursive_v2",
            "residual_block_gated_v2",
            "sagn_lite_v3",
            "gamlp_lite_v3",
        ] = "sagn_lite_v2",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
        block_dropout: float = 0.0,
        hop_dropout: float = 0.0,
        label_branch_hidden: int = 128,
        label_dropout: float = 0.0,
        attention_heads: int = 1,
        activation: str = "relu",
        norm: str = "none",
    ) -> None:
        super().__init__()
        del label_branch_hidden
        if not block_dims:
            raise ValueError("at least one SFT block is required")
        forbidden = [name for name in block_dims if _contains_forbidden_logit_name(name)]
        if forbidden:
            raise ValueError(f"logits are forbidden as T2.2 SFT-NL input blocks: {forbidden}")
        self.block_dims = OrderedDict((str(name), int(dim)) for name, dim in block_dims.items())
        aliases = {"sagn_lite_v3": "sagn_lite_v2", "gamlp_lite_v3": "gamlp_lite_v2"}
        self.model_type = str(model_type)
        canonical_model_type = aliases.get(self.model_type, self.model_type)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.label_dropout = float(label_dropout)
        self.attention_heads = int(attention_heads)
        self.label_blocks = [name for name in self.block_dims if name.lower().startswith("y") or name.lower().startswith("label")]
        self.structure_blocks = [name for name in self.block_dims if name == "structure" or name.lower().startswith("degree")]
        self.normalizers = nn.ModuleDict({name: _TorchBlockStandardizer(dim, name=name) for name, dim in self.block_dims.items()})
        if canonical_model_type == "sagn_lite_v2":
            self.core = SAGNLiteV2Core(
                self.block_dims,
                hidden_dim=hidden_dim,
                dropout=dropout,
                block_dropout=block_dropout,
                hop_dropout=hop_dropout,
                num_layers=num_layers,
                activation=activation,
                norm=norm,
            )
        elif canonical_model_type == "gamlp_lite_v2":
            self.core = GAMLPLiteCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        elif canonical_model_type == "gamlp_recursive_v2":
            self.core = GAMLPRecursiveV2Core(self.block_dims, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers, activation=activation, norm=norm)
        elif canonical_model_type == "residual_block_gated_v2":
            self.core = ResidualBlockGatedCore(self.block_dims, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError("unsupported T23 SFT teacher model_type")
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
        out = {name: self.normalizers[name](blocks[name].to(torch.float32)) for name in self.block_dims}
        if self.training and self.label_dropout > 0.0:
            keep_prob = max(1e-12, 1.0 - float(self.label_dropout))
            for name in self.label_blocks:
                mask = (torch.rand(out[name].shape, device=out[name].device) < keep_prob).to(out[name].dtype)
                out[name] = out[name] * mask / keep_prob
        return out

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
            "has_label_branch": bool(self.label_blocks),
            "has_structure_branch": bool(self.structure_blocks),
            "label_dropout": float(self.label_dropout),
            "attention_heads": int(self.attention_heads),
            "label_blocks": list(self.label_blocks),
            "structure_blocks": list(self.structure_blocks),
            "uses_logits_as_input": False,
            "uses_teacher_logits": False,
            "uses_kd": False,
            "uses_full_graph_backprop": False,
            "final_logits_activation": "none",
        }
        if hasattr(self.core, "attention_mean"):
            diag["attention_mean"] = self.core.attention_mean()
        return diag


SFTTableTeacherV3 = SFTTeacherV3
