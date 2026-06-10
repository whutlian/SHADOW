from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from shadow_hgc.eval.sft_eval import predict_sft_logits, sft_metrics
from shadow_hgc.models.sft_table_teacher import SFTTableTeacherV2 as SFTTableTeacher


@dataclass
class SFTTeacherTrainResult:
    model: SFTTableTeacher
    logits: torch.Tensor
    summary: dict[str, Any]


def sft_loss(logits: torch.Tensor, labels: torch.Tensor, *, loss_type: str, train_labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    loss_type = str(loss_type)
    if loss_type == "cross_entropy":
        return F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
    if loss_type == "class_balanced_ce":
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=float(label_smoothing))
        counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
        weights = (1.0 / counts)[labels]
        return (weights * ce).sum() / weights.sum().clamp_min(1e-12)
    if loss_type == "balanced_softmax":
        counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
        return F.cross_entropy(logits + counts.log().unsqueeze(0), labels, label_smoothing=float(label_smoothing))
    if loss_type in {"logit_adjusted_ce", "logit_adjusted_ce_as_training_loss_only", "logit_adjusted_ce_as_loss"}:
        counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
        prior = counts / counts.sum()
        return F.cross_entropy(logits - prior.log().unsqueeze(0), labels, label_smoothing=float(label_smoothing))
    if loss_type == "label_smoothing_ce":
        smoothing = float(label_smoothing) if label_smoothing > 0 else 0.05
        return F.cross_entropy(logits, labels, label_smoothing=smoothing)
    if loss_type == "focal_loss":
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=float(label_smoothing))
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        gamma = 2.0
        return (((1.0 - pt) ** gamma) * ce).mean()
    if loss_type == "sqrt_weighted_ce":
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=float(label_smoothing))
        counts = torch.bincount(train_labels, minlength=logits.shape[1]).to(logits.device, logits.dtype).clamp_min(1.0)
        weights = (1.0 / torch.sqrt(counts))[labels]
        return (weights * ce).sum() / weights.sum().clamp_min(1e-12)
    raise ValueError(f"unsupported SFT loss type: {loss_type}")


def _loss(logits: torch.Tensor, labels: torch.Tensor, *, loss_type: str, train_labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return sft_loss(logits, labels, loss_type=loss_type, train_labels=train_labels, label_smoothing=label_smoothing)


def train_sft_teacher(
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    test_rows: torch.Tensor,
    *,
    num_classes: int,
    model_type: str = "sagn_lite",
    hidden_dim: int = 256,
    dropout: float = 0.3,
    num_layers_per_block: int = 1,
    loss_type: str = "cross_entropy",
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 30,
    seed: int = 42,
    batch_size: int | None = None,
    label_smoothing: float = 0.0,
) -> SFTTeacherTrainResult:
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    block_dims = {name: int(value.shape[1]) for name, value in blocks.items()}
    model = SFTTableTeacher(
        block_dims,
        num_classes=int(num_classes),
        model_type=model_type,  # type: ignore[arg-type]
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        num_layers_per_block=int(num_layers_per_block),
    )
    model.fit_block_stats(blocks, train_rows=train_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    labels = labels.to(torch.long)
    train_labels = labels[train_rows].to(torch.long)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        if batch_size is None:
            opt.zero_grad(set_to_none=True)
            logits = model(blocks)
            loss = _loss(logits[train_rows], train_labels, loss_type=loss_type, train_labels=train_labels, label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
        else:
            order = train_rows[torch.randperm(train_rows.numel())]
            for start in range(0, int(order.numel()), int(batch_size)):
                rows = order[start : start + int(batch_size)]
                opt.zero_grad(set_to_none=True)
                batch_blocks = {name: value[rows] for name, value in blocks.items()}
                logits = model(batch_blocks)
                loss = _loss(logits, labels[rows], loss_type=loss_type, train_labels=train_labels, label_smoothing=label_smoothing)
                loss.backward()
                opt.step()
        with torch.no_grad():
            val_logits = model({name: value[valid_rows] for name, value in blocks.items()})
        val_metrics = sft_metrics(val_logits, labels[valid_rows], torch.arange(valid_rows.numel()), num_classes=int(num_classes))
        improves = val_metrics["accuracy"] > best_val_acc or (
            abs(val_metrics["accuracy"] - best_val_acc) <= 1e-12 and val_metrics["macro_f1"] > best_val_f1
        )
        if improves:
            best_val_acc = val_metrics["accuracy"]
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    logits = predict_sft_logits(model, blocks, batch_size=batch_size)
    summary = {
        "model_type": model_type,
        "loss_type": loss_type,
        "seed": int(seed),
        "epochs_ran": int(epoch + 1),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(best_val_f1),
        "train": sft_metrics(logits, labels, train_rows, num_classes=int(num_classes)),
        "valid": sft_metrics(logits, labels, valid_rows, num_classes=int(num_classes)),
        "test": sft_metrics(logits, labels, test_rows, num_classes=int(num_classes)),
        "training_time_s": float(time.perf_counter() - started),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_full_graph_backprop": False,
        **model.diagnostics(),
    }
    return SFTTeacherTrainResult(model=model, logits=logits, summary=summary)
