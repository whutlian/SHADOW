from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.models.safe_block_fusion import SafeBlockFusionClassifier


@dataclass
class SafeBlockSelectionResult:
    selected_blocks: list[str]
    block_diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: SafeBlockFusionClassifier


def _metrics(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, num_classes: int) -> dict[str, Any]:
    pred = logits.argmax(dim=1).to(torch.long)
    selected = pred[rows]
    y = labels[rows].to(torch.long)
    return {
        "accuracy": float((selected == y).to(torch.float32).mean().item()) if rows.numel() else 0.0,
        "macro_f1": macro_f1_score(selected, y, num_classes=num_classes),
        "predicted_class_count": int((torch.bincount(selected.clamp_min(0), minlength=num_classes) > 0).sum().item()),
    }


def _slice(blocks: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[rows] for name, value in blocks.items()}


def _train_model(
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
) -> SafeBlockFusionClassifier:
    torch.manual_seed(int(seed))
    model = SafeBlockFusionClassifier({name: int(value.shape[1]) for name, value in blocks.items()}, num_classes=num_classes, hidden_dim=hidden_dim)
    model.fit_block_stats(_slice(blocks, train_rows), source="train_target_rows")
    gate_params = [model.raw_gates]
    gate_param_ids = {id(param) for param in gate_params}
    other_params = [param for param in model.parameters() if id(param) not in gate_param_ids]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": float(lr), "weight_decay": float(weight_decay)},
            {"params": gate_params, "lr": float(lr) * 100.0, "weight_decay": 0.0},
        ]
    )
    y = labels[train_rows].to(torch.long)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(_slice(blocks, train_rows))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return model


def _block_norm_stats_source(block: torch.Tensor) -> dict[str, float]:
    x = block.detach().to(torch.float32)
    return {
        "mean_abs": float(x.abs().mean().item()),
        "std_mean": float(x.std(dim=0, unbiased=False).mean().item()) if x.numel() else 0.0,
    }


def train_with_validation_gated_blocks(
    blocks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    train_rows: torch.Tensor,
    val_rows: torch.Tensor,
    test_rows: torch.Tensor,
    num_classes: int,
    seed: int = 42,
    epochs: int = 120,
    hidden_dim: int = 64,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    epsilon_acc: float = 0.001,
    epsilon_f1: float = 0.002,
) -> SafeBlockSelectionResult:
    if "self" not in blocks:
        raise ValueError("blocks must contain self")
    self_blocks = {"self": blocks["self"]}
    self_model = _train_model(
        self_blocks,
        labels,
        train_rows,
        num_classes=num_classes,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
    )
    with torch.no_grad():
        self_logits = self_model(self_blocks)
    self_val = _metrics(self_logits, labels, val_rows, num_classes)
    self_test = _metrics(self_logits, labels, test_rows, num_classes)
    kept = ["self"]
    diagnostics: list[dict[str, Any]] = []
    for index, block_name in enumerate(name for name in blocks if name != "self"):
        candidate_blocks = {"self": blocks["self"], block_name: blocks[block_name]}
        model = _train_model(
            candidate_blocks,
            labels,
            train_rows,
            num_classes=num_classes,
            seed=seed + index + 1,
            epochs=epochs,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
        )
        with torch.no_grad():
            logits = model(candidate_blocks)
        safe_val_metrics = _metrics(logits, labels, val_rows, num_classes)
        branch_probe_blocks = {"self": blocks[block_name]}
        branch_probe = _train_model(
            branch_probe_blocks,
            labels,
            train_rows,
            num_classes=num_classes,
            seed=seed + index + 1009,
            epochs=epochs,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
        )
        with torch.no_grad():
            branch_logits = branch_probe(branch_probe_blocks)
        train_metrics = _metrics(branch_logits, labels, train_rows, num_classes)
        val_metrics = _metrics(branch_logits, labels, val_rows, num_classes)
        test_metrics = _metrics(branch_logits, labels, test_rows, num_classes)
        gate_initial = float(torch.nn.functional.softplus(torch.tensor(-8.0)).item())
        gate_final = model.gate_values()[block_name]
        improves = (
            val_metrics["accuracy"] - self_val["accuracy"] >= float(epsilon_acc)
            or val_metrics["macro_f1"] - self_val["macro_f1"] >= float(epsilon_f1)
        )
        safe_non_regression = (
            safe_val_metrics["accuracy"] >= self_val["accuracy"] - 1e-12
            or safe_val_metrics["macro_f1"] >= self_val["macro_f1"] - 1e-12
        )
        decision = "kept" if improves and safe_non_regression else "dropped"
        if decision == "kept":
            kept.append(block_name)
        diagnostics.append(
            {
                "block_name": block_name,
                "block_type": block_name.split(":", 1)[0],
                "block_dim": int(blocks[block_name].shape[1]),
                "branch_train_acc": train_metrics["accuracy"],
                "branch_val_acc": val_metrics["accuracy"],
                "branch_test_acc": test_metrics["accuracy"],
                "branch_val_macro_f1": val_metrics["macro_f1"],
                "safe_candidate_val_acc": safe_val_metrics["accuracy"],
                "safe_candidate_val_macro_f1": safe_val_metrics["macro_f1"],
                "gate_initial": gate_initial,
                "gate_final": gate_final,
                "kept_or_dropped": decision,
                "drop_reason": "" if decision == "kept" else "validation_non_regression_gate",
                "block_norm_stats_source": _block_norm_stats_source(blocks[block_name]),
                "block_nan_count": int(torch.isnan(blocks[block_name]).sum().item()),
            }
        )
    final_blocks = {name: blocks[name] for name in kept}
    final_model = _train_model(
        final_blocks,
        labels,
        train_rows,
        num_classes=num_classes,
        seed=seed + 97,
        epochs=epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
    )
    with torch.no_grad():
        final_logits = final_model(final_blocks)
    final_val = _metrics(final_logits, labels, val_rows, num_classes)
    final_test = _metrics(final_logits, labels, test_rows, num_classes)
    summary = {
        "model_type": "safe_block_fusion",
        "selected_blocks": kept,
        "self_val_acc": self_val["accuracy"],
        "self_val_macro_f1": self_val["macro_f1"],
        "self_test_acc": self_test["accuracy"],
        "final_val_acc": final_val["accuracy"],
        "final_val_macro_f1": final_val["macro_f1"],
        "final_test_acc": final_test["accuracy"],
        "final_test_macro_f1": final_test["macro_f1"],
        "final_predicted_class_count": final_test["predicted_class_count"],
        **final_model.diagnostics(),
    }
    return SafeBlockSelectionResult(selected_blocks=kept, block_diagnostics=diagnostics, summary=summary, model=final_model)
