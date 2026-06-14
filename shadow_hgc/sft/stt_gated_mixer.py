from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from shadow_hgc.models.sagn_lite import make_mlp
from shadow_hgc.models.sft_teacher import _TorchBlockStandardizer, _contains_forbidden_logit_name
from shadow_hgc.sft.unified_schedule import compute_unified_schedule


@dataclass(frozen=True)
class STTGatedMixerConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int
    epochs: int
    internal_style: str
    dropout: float = 0.10


class STTGatedMixer(nn.Module):
    """Unified block-wise table student for the public STT-U path.

    The model consumes graph-signal table blocks only. Teacher probabilities are
    not accepted as input features; teacher information can affect selection and
    loss weights outside this module.
    """

    def __init__(
        self,
        block_dims: dict[str, int],
        *,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        internal_style: str = "gamlp_like",
        num_layers: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ) -> None:
        super().__init__()
        if not block_dims:
            raise ValueError("at least one graph-signal block is required")
        forbidden = [name for name in block_dims if _contains_forbidden_logit_name(str(name))]
        if forbidden:
            raise ValueError(f"teacher/logit blocks are forbidden as STT-GatedMixer inputs: {forbidden}")
        self.block_dims = OrderedDict((str(name), int(dim)) for name, dim in block_dims.items())
        self.block_names = list(self.block_dims)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.internal_style = str(internal_style)
        self.normalizers = nn.ModuleDict({name: _TorchBlockStandardizer(dim, name=name) for name, dim in self.block_dims.items()})
        self.projections = nn.ModuleDict(
            {
                name: make_mlp(
                    dim,
                    int(hidden_dim),
                    num_layers=max(1, int(num_layers)),
                    dropout=float(dropout),
                    activation=str(activation),
                    norm=str(norm),
                )
                for name, dim in self.block_dims.items()
            }
        )
        self.raw_block_gates = nn.Parameter(torch.zeros(len(self.block_names)))
        self.raw_label_gate = nn.Parameter(torch.tensor(0.0))
        self.raw_structure_gate = nn.Parameter(torch.tensor(0.0))
        self.residual = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.mixer_norm = nn.LayerNorm(int(hidden_dim))
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
        self.label_blocks = [name for name in self.block_names if name.lower().startswith("y") or name.lower().startswith("label")]
        self.structure_blocks = [name for name in self.block_names if name == "structure" or name.lower().startswith("degree")]

    def _check(self, blocks: dict[str, torch.Tensor]) -> None:
        for name, dim in self.block_dims.items():
            if name not in blocks:
                raise ValueError(f"missing graph-signal block {name}")
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

    def _gate_multiplier(self, name: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = torch.tensor(1.0, device=device, dtype=dtype)
        if name in self.label_blocks:
            value = value * torch.sigmoid(self.raw_label_gate.to(device=device, dtype=dtype))
        if name in self.structure_blocks:
            value = value * torch.sigmoid(self.raw_structure_gate.to(device=device, dtype=dtype))
        return value

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check(blocks)
        normalized = self._normalized(blocks)
        first = normalized[self.block_names[0]]
        gates = torch.sigmoid(self.raw_block_gates.to(device=first.device, dtype=first.dtype))
        hidden = None
        for idx, name in enumerate(self.block_names):
            branch = self.projections[name](normalized[name].to(device=first.device, dtype=first.dtype))
            gate = gates[idx] * self._gate_multiplier(name, device=branch.device, dtype=branch.dtype)
            hidden = gate * branch if hidden is None else hidden + gate * branch
        assert hidden is not None
        hidden = self.mixer_norm(hidden + self.residual(hidden))
        return self.classifier(hidden)

    def gate_values(self) -> dict[str, float]:
        block = torch.sigmoid(self.raw_block_gates.detach().cpu())
        values = {name: float(block[idx].item()) for idx, name in enumerate(self.block_names)}
        values["label_reuse"] = float(torch.sigmoid(self.raw_label_gate.detach().cpu()).item())
        values["structure"] = float(torch.sigmoid(self.raw_structure_gate.detach().cpu()).item())
        return values

    def block_norm_metadata(self) -> dict:
        return {name: self.normalizers[name].metadata() for name in self.block_dims}

    def diagnostics(self) -> dict:
        return {
            "model_type": "stt_gated_mixer",
            "student_family": "stt_gated_mixer",
            "student_internal_style": self.internal_style,
            "block_dims": dict(self.block_dims),
            "block_gates": self.gate_values(),
            "block_norm_stats": self.block_norm_metadata(),
            "block_norm_stats_source": "train_target_rows",
            "has_label_reuse_gates": bool(self.label_blocks),
            "has_structure_gates": bool(self.structure_blocks),
            "label_blocks": list(self.label_blocks),
            "structure_blocks": list(self.structure_blocks),
            "uses_logits_as_input": False,
            "uses_teacher_logits": False,
            "uses_kd": False,
            "uses_full_graph_backprop": False,
            "final_logits_activation": "none",
        }


def make_stt_gated_mixer_config(
    *,
    input_dim: int,
    num_classes: int,
    condensed_nodes: int,
    teacher_valid_acc: float | None = None,
    majority_valid_acc: float | None = None,
    num_nodes: int = 1,
) -> STTGatedMixerConfig:
    schedule = compute_unified_schedule(
        condensed_nodes=int(condensed_nodes),
        num_classes=int(num_classes),
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=majority_valid_acc,
        num_nodes=int(num_nodes),
        num_teacher_nodes=int(num_nodes),
    )
    return STTGatedMixerConfig(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        hidden_dim=schedule.hidden_dim,
        epochs=schedule.epochs,
        internal_style=schedule.student_internal_style,
    )


def estimate_stt_gated_mixer_params(*, input_dim: int, num_classes: int, hidden_dim: int) -> int:
    # Gated table mixer proxy used for reporting: input projection, two gates, classifier, and biases.
    return int(input_dim) * int(hidden_dim) * 3 + int(hidden_dim) * int(num_classes) + int(hidden_dim) * 3 + int(num_classes)
