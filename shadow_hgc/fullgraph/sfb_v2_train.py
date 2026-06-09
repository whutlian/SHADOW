from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.fullgraph.sfb_v2_infer import predict_sfb_v2_logits
from shadow_hgc.models.block_gated_table import BlockGatedTableModel


@dataclass
class SFBV2TrainResult:
    model: BlockGatedTableModel
    summary: dict
    logits: torch.Tensor


def should_run_medium_row(*, dataset: str, estimated_cache_bytes: int, memory_limit_bytes: int) -> dict:
    if int(estimated_cache_bytes) <= int(memory_limit_bytes):
        return {"dataset": dataset, "should_run": True, "status": "run_allowed", "reason": "estimate_under_limit"}
    return {"dataset": dataset, "should_run": False, "status": "blocked_resource_guard", "reason": "estimate_exceeds_limit"}


def format_allocation_failure(
    *,
    tensor_shape: tuple[int, ...],
    requested_bytes: int,
    chunk_size: int,
    current_cache_bytes: int,
    peak_ram_gb: float,
    module_name: str,
) -> dict:
    return {
        "tensor_shape": [int(value) for value in tensor_shape],
        "requested_bytes": int(requested_bytes),
        "chunk_size": int(chunk_size),
        "current_cache_bytes": int(current_cache_bytes),
        "peak_ram_gb": float(peak_ram_gb),
        "module_name": str(module_name),
    }


def _weighted_f1(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    total = 0.0
    score = 0.0
    y = labels[idx]
    p = pred[idx]
    for class_id in range(int(num_classes)):
        true_mask = y == class_id
        pred_mask = p == class_id
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
    return score / total if total else 0.0


def _metrics(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, num_classes: int) -> dict:
    pred = logits.argmax(dim=1).to(torch.long)
    selected = pred[rows]
    y = labels[rows].to(torch.long)
    hist = torch.bincount(selected.clamp_min(0), minlength=int(num_classes)).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if hist.numel() else 0.0
    return {
        "accuracy": float((selected == y).to(torch.float32).mean().item()) if y.numel() else 0.0,
        "macro_f1": macro_f1_score(selected, y, num_classes=num_classes),
        "weighted_f1": _weighted_f1(pred, labels, rows, num_classes),
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
    }


def train_sfb_v2_table_model(
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    val_rows: torch.Tensor,
    test_rows: torch.Tensor,
    *,
    num_classes: int,
    seed: int = 42,
    epochs: int = 200,
    patience: int = 40,
    hidden_dim: int = 256,
    branch_dropout: float = 0.3,
    lr: float = 0.003,
    weight_decay: float = 5e-4,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    batch_size: int | None = None,
) -> SFBV2TrainResult:
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    block_dims = {name: int(value.shape[1]) for name, value in blocks.items()}
    model = BlockGatedTableModel(block_dims, num_classes=num_classes, hidden_dim=hidden_dim, branch_dropout=branch_dropout)
    model.fit_block_stats({name: value[train_rows] for name, value in blocks.items()}, source="train_target_rows")
    model.freeze_block_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        if batch_size is None:
            opt.zero_grad(set_to_none=True)
            logits = model(blocks)
            train_logits = logits[train_rows]
            train_labels = labels[train_rows].to(torch.long)
            if loss_type == "balanced_softmax":
                counts = torch.bincount(train_labels, minlength=num_classes).to(train_logits.device, train_logits.dtype).clamp_min(1.0)
                loss = F.cross_entropy(train_logits + counts.log().unsqueeze(0), train_labels, label_smoothing=label_smoothing)
            elif loss_type == "logit_adjusted":
                counts = torch.bincount(train_labels, minlength=num_classes).to(train_logits.device, train_logits.dtype).clamp_min(1.0)
                prior = counts / counts.sum()
                loss = F.cross_entropy(train_logits - prior.log().unsqueeze(0), train_labels, label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(train_logits, train_labels, label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
        else:
            order = train_rows[torch.randperm(train_rows.numel())]
            for start in range(0, int(order.numel()), int(batch_size)):
                batch_rows = order[start : start + int(batch_size)]
                batch_blocks = {name: value[batch_rows] for name, value in blocks.items()}
                opt.zero_grad(set_to_none=True)
                train_logits = model(batch_blocks)
                train_labels = labels[batch_rows].to(torch.long)
                if loss_type == "balanced_softmax":
                    counts = torch.bincount(labels[train_rows].to(torch.long), minlength=num_classes).to(train_logits.device, train_logits.dtype).clamp_min(1.0)
                    loss = F.cross_entropy(train_logits + counts.log().unsqueeze(0), train_labels, label_smoothing=label_smoothing)
                elif loss_type == "logit_adjusted":
                    counts = torch.bincount(labels[train_rows].to(torch.long), minlength=num_classes).to(train_logits.device, train_logits.dtype).clamp_min(1.0)
                    prior = counts / counts.sum()
                    loss = F.cross_entropy(train_logits - prior.log().unsqueeze(0), train_labels, label_smoothing=label_smoothing)
                else:
                    loss = F.cross_entropy(train_logits, train_labels, label_smoothing=label_smoothing)
                loss.backward()
                opt.step()
        with torch.no_grad():
            if batch_size is None:
                val_logits = model(blocks)
                val_acc = _metrics(val_logits, labels, val_rows, num_classes)["accuracy"]
            else:
                val_blocks = {name: value[val_rows] for name, value in blocks.items()}
                val_logits = model(val_blocks)
                pred = val_logits.argmax(dim=1).to(torch.long)
                val_acc = float((pred == labels[val_rows].to(torch.long)).to(torch.float32).mean().item()) if val_rows.numel() else 0.0
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    logits = predict_sfb_v2_logits(model, blocks, batch_size=batch_size)
    train_metrics = _metrics(logits, labels, train_rows, num_classes)
    val_metrics = _metrics(logits, labels, val_rows, num_classes)
    test_metrics = _metrics(logits, labels, test_rows, num_classes)
    summary: dict[str, Any] = {
        "model_type": "sfb_v2",
        "enabled_blocks": list(blocks),
        "epochs_ran": int(epoch + 1),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val),
        "self_train_acc": train_metrics["accuracy"],
        "self_val_acc": val_metrics["accuracy"],
        "self_test_acc": test_metrics["accuracy"],
        "self_predicted_class_count": test_metrics["predicted_class_count"],
        "self_macro_f1": test_metrics["macro_f1"],
        "test_acc": test_metrics["accuracy"],
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "predicted_class_count": test_metrics["predicted_class_count"],
        "prediction_entropy": test_metrics["prediction_entropy"],
        "training_time_s": float(time.perf_counter() - started),
        "inference_time_s": 0.0,
        **model.diagnostics(),
    }
    return SFBV2TrainResult(model=model, summary=summary, logits=logits)
