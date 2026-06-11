from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class QOCTransferResult:
    model: nn.Module
    logits_real: torch.Tensor
    metrics: dict[str, Any]


class QOCTableHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))


def _macro_f1(pred: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    encoded = labels.to(torch.long).cpu().clamp_min(0) * int(num_classes) + pred.to(torch.long).cpu().clamp_min(0)
    confusion = torch.bincount(encoded, minlength=int(num_classes) * int(num_classes)).view(int(num_classes), int(num_classes))
    tp = torch.diag(confusion).to(torch.float64)
    fp = confusion.sum(dim=0).to(torch.float64) - tp
    fn = confusion.sum(dim=1).to(torch.float64) - tp
    return float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-12)).mean().item())


def train_qoc_table_head(
    *,
    input_syn: torch.Tensor,
    labels_syn: torch.Tensor,
    code_weights: torch.Tensor,
    input_real: torch.Tensor,
    labels_real: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 128,
    epochs: int = 120,
    lr: float = 0.05,
    dropout: float = 0.0,
    seed: int = 42,
    eval_rows: torch.Tensor | None = None,
) -> QOCTransferResult:
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    model = QOCTableHead(int(input_syn.shape[1]), int(num_classes), hidden_dim=int(hidden_dim), dropout=float(dropout))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_syn = input_syn.to(torch.float32)
    y_syn = labels_syn.to(torch.long)
    weights = code_weights.to(torch.float32)
    weights = weights / weights.mean().clamp_min(1e-12)
    train_started = time.perf_counter()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad()
        logits = model(x_syn)
        per = torch.nn.functional.cross_entropy(logits, y_syn, reduction="none")
        loss = (per * weights).mean()
        loss.backward()
        optimizer.step()
    training_time = time.perf_counter() - train_started
    eval_started = time.perf_counter()
    with torch.no_grad():
        eval_input = input_real.to(torch.float32)
        eval_labels = labels_real.to(torch.long)
        if eval_rows is not None:
            rows = eval_rows.to(torch.long).cpu()
            eval_input = eval_input[rows]
            eval_labels = eval_labels[rows]
        logits_real = model(eval_input)
        pred = logits_real.argmax(dim=1)
        acc = float((pred == eval_labels).to(torch.float32).mean().item())
        macro = _macro_f1(pred, eval_labels, int(num_classes))
    eval_time = time.perf_counter() - eval_started
    metrics = {
        "accuracy": acc,
        "macro_f1": macro,
        "predicted_classes": int(torch.unique(pred).numel()),
        "eval_rows": int(eval_labels.numel()),
        "transfer_eval_type": "real_transfer_eval",
        "student_model": "operator_sft_table_head",
        "training_time": float(training_time),
        "eval_time": float(eval_time),
        "condensation_time": float(time.perf_counter() - started),
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_adjacency": False,
        "uses_e_by_d_materialization": False,
    }
    return QOCTransferResult(model=model, logits_real=logits_real, metrics=metrics)
