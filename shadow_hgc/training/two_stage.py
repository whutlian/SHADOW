from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.eval.sft_eval import predict_sft_logits, sft_metrics
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.train.train_sft_teacher import sft_loss


@dataclass(frozen=True)
class StageConfig:
    loss_type: str
    epochs: int
    lr_mult: float


@dataclass(frozen=True)
class TwoStageConfig:
    enabled: bool = True
    stage1_loss: str = "sqrt_weighted_ce"
    stage2_loss: str = "cross_entropy"
    stage1_epochs: int = 100
    stage2_epochs: int = 100
    stage2_lr_mult: float = 0.2

    def stages(self) -> list[StageConfig]:
        if not self.enabled:
            return [StageConfig(loss_type=self.stage2_loss, epochs=self.stage2_epochs, lr_mult=1.0)]
        return [
            StageConfig(loss_type=self.stage1_loss, epochs=int(self.stage1_epochs), lr_mult=1.0),
            StageConfig(loss_type=self.stage2_loss, epochs=int(self.stage2_epochs), lr_mult=float(self.stage2_lr_mult)),
        ]


@dataclass(frozen=True)
class TwoStageTrainResult:
    model: SFTTeacherV3
    logits: torch.Tensor
    summary: dict[str, Any]


def _slice(blocks: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[rows] for name, value in blocks.items()}


def _iter_batches(rows: torch.Tensor, *, batch_size: int | None, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    order = rows[torch.randperm(rows.numel(), generator=generator)]
    if batch_size is None:
        return [order]
    return [order[start : start + int(batch_size)] for start in range(0, int(order.numel()), int(batch_size))]


def train_sft_two_stage(
    *,
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    test_rows: torch.Tensor,
    num_classes: int,
    model_type: str = "sagn_lite_v2",
    hidden_dim: int = 512,
    dropout: float = 0.3,
    num_layers: int = 2,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    config: TwoStageConfig | None = None,
    batch_size: int | None = 16_384,
    seed: int = 42,
    label_smoothing: float = 0.0,
) -> TwoStageTrainResult:
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    config = config or TwoStageConfig()
    labels = labels.to(torch.long)
    train_rows = train_rows.to(torch.long)
    valid_rows = valid_rows.to(torch.long)
    test_rows = test_rows.to(torch.long)
    model = SFTTeacherV3(
        {name: int(value.shape[1]) for name, value in blocks.items()},
        num_classes=int(num_classes),
        model_type=model_type,  # type: ignore[arg-type]
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        num_layers=int(num_layers),
    )
    model.fit_block_stats(blocks, train_rows=train_rows)
    train_labels = labels[train_rows]
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    stage_summaries: list[dict[str, Any]] = []
    epoch_offset = 0
    for stage in config.stages():
        for group in opt.param_groups:
            group["lr"] = float(lr) * float(stage.lr_mult)
        last_loss = 0.0
        seen = 0
        for epoch in range(int(stage.epochs)):
            model.train()
            for rows in _iter_batches(train_rows, batch_size=batch_size, seed=int(seed) + epoch_offset + epoch):
                opt.zero_grad(set_to_none=True)
                logits = model(_slice(blocks, rows))
                loss_type = "logit_adjusted_ce_as_training_loss_only" if stage.loss_type == "logit_adjusted_ce_as_loss" else stage.loss_type
                loss = sft_loss(logits, labels[rows], loss_type=loss_type, train_labels=train_labels, label_smoothing=label_smoothing)
                loss.backward()
                opt.step()
                last_loss += float(loss.detach().cpu().item()) * int(rows.numel())
                seen += int(rows.numel())
        with torch.no_grad():
            valid_logits = model(_slice(blocks, valid_rows))
        valid = sft_metrics(valid_logits, labels[valid_rows], torch.arange(valid_rows.numel()), num_classes=int(num_classes))
        stage_summaries.append(
            {
                "loss_type": stage.loss_type,
                "epochs": int(stage.epochs),
                "lr_mult": float(stage.lr_mult),
                "train_loss": last_loss / max(1, seen),
                "valid": valid,
            }
        )
        epoch_offset += int(stage.epochs)
    logits = predict_sft_logits(model, blocks, batch_size=batch_size)
    summary = {
        "model_type": model_type,
        "stages": stage_summaries,
        "epochs_ran": sum(int(stage.epochs) for stage in config.stages()),
        "train": sft_metrics(logits, labels, train_rows, num_classes=int(num_classes)),
        "valid": sft_metrics(logits, labels, valid_rows, num_classes=int(num_classes)),
        "test": sft_metrics(logits, labels, test_rows, num_classes=int(num_classes)),
        "training_time_s": float(time.perf_counter() - started),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        **model.diagnostics(),
    }
    return TwoStageTrainResult(model=model, logits=logits, summary=summary)
