from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def zero_predicted_classes(pred: torch.Tensor, *, num_classes: int) -> int:
    labels = pred.detach().cpu().long().view(-1)
    if labels.numel() == 0:
        return int(num_classes)
    counts = torch.bincount(labels.clamp_min(0), minlength=int(num_classes))
    return int((counts[: int(num_classes)] == 0).sum().item())


def class_coverage_loss(probs: torch.Tensor, target_prior: torch.Tensor) -> torch.Tensor:
    p = probs.detach().float()
    if p.ndim != 2:
        raise ValueError("probs must be rank-2")
    p = p.clamp_min(0.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    pred_prior = p.mean(dim=0).clamp_min(1e-12)
    target = target_prior.detach().float().to(pred_prior.device)
    target = target / target.sum().clamp_min(1e-12)
    return F.kl_div(pred_prior.log(), target, reduction="sum")


def products_promotion_status(
    *,
    method: str,
    ratio: float,
    accuracy: float,
    macro_f1: float,
    predicted_classes: int,
) -> tuple[str, str]:
    name = str(method)
    ratio = float(ratio)
    acc = float(accuracy)
    macro = float(macro_f1)
    pred = int(predicted_classes)
    if "balanced" in name:
        accuracy_ok = acc >= (0.735 if abs(ratio - 0.0025) < 1e-12 else 0.0)
        if pred >= 40 and macro > 0.400 and accuracy_ok:
            return "promoted", ""
        return "not_promoted", "products_balanced_macro_or_class_gate_not_met"
    if "official" in name:
        if acc >= 0.800:
            return "promoted", ""
        return "not_promoted", "products_official_accuracy_gate_not_met"
    return "not_promoted", "products_method_not_in_t34_gate"


def products_row_from_source(source: dict[str, Any]) -> dict[str, Any]:
    required = ["source_csv", "source_method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes"]
    missing = [field for field in required if source.get(field) in {"", None}]
    if missing:
        return {"status": "blocked", "failure_reason": "missing_products_source_fields:" + ",".join(missing)}
    if str(source.get("status", "")) != "completed_long":
        return {"status": "blocked", "failure_reason": "products_source_not_completed_long"}
    return dict(source)
