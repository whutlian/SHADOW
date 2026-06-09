from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.fullgraph.sfb_infer import evaluate_logits, predict_logits
from shadow_hgc.fullgraph.sfb_model import BlockGatedResidualTableModel


@dataclass
class SFBTrainResult:
    model: BlockGatedResidualTableModel
    summary: dict
    logits: torch.Tensor


def _to_tensor_rows(rows: torch.Tensor | list[int]) -> torch.Tensor:
    if isinstance(rows, torch.Tensor):
        return rows.long()
    return torch.tensor(rows, dtype=torch.long)


def _weighted_f1(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    selected_pred = pred[idx]
    selected_labels = labels[idx]
    total = 0.0
    score = 0.0
    for class_id in range(int(num_classes)):
        true_mask = selected_labels == class_id
        pred_mask = selected_pred == class_id
        support = float(true_mask.sum().item())
        if support == 0:
            continue
        tp = float((true_mask & pred_mask).sum().item())
        fp = float((~true_mask & pred_mask).sum().item())
        fn = float((true_mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        score += support * f1
        total += support
    return score / total if total > 0 else 0.0


def train_sfb_table_model(
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor | list[int],
    val_rows: torch.Tensor | list[int] | None,
    test_rows: torch.Tensor | list[int],
    *,
    num_classes: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    block_dropout: float = 0.0,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 40,
    seed: int = 42,
    fusion: str = "residual_logits",
) -> SFBTrainResult:
    torch.manual_seed(int(seed))
    train_idx = _to_tensor_rows(train_rows)
    val_idx = _to_tensor_rows(val_rows) if val_rows is not None else train_idx
    test_idx = _to_tensor_rows(test_rows)
    block_dims = {name: int(value.shape[1]) for name, value in blocks.items()}
    model = BlockGatedResidualTableModel(
        block_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        block_dropout=block_dropout,
        fusion=fusion,  # type: ignore[arg-type]
    )
    model.fit_block_stats({name: value[train_idx] for name, value in blocks.items()}, source="train_target_rows")
    model.freeze_block_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(blocks)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_logits = model(blocks)
            val_acc = evaluate_logits(val_logits, labels, val_idx)["accuracy"]
        if float(val_acc) > best_val:
            best_val = float(val_acc)
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    logits = predict_logits(model, blocks)
    pred = logits.argmax(dim=1).to(torch.long)
    train_metrics = evaluate_logits(logits, labels, train_idx)
    val_metrics = evaluate_logits(logits, labels, val_idx)
    test_metrics = evaluate_logits(logits, labels, test_idx)
    summary = {
        "model_type": "sfb",
        "epochs_ran": int(epoch + 1),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val),
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "accuracy": test_metrics["accuracy"],
        "macro_f1": macro_f1_score(pred[test_idx], labels[test_idx], num_classes=num_classes),
        "weighted_f1": _weighted_f1(pred, labels, test_idx, num_classes),
        "predicted_class_count": test_metrics["predicted_class_count"],
        "training_time_s": float(time.perf_counter() - started),
        **model.diagnostics(),
    }
    return SFBTrainResult(model=model, summary=summary, logits=logits)
