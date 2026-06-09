from __future__ import annotations

from itertools import product
from typing import Iterable

import torch


def _round_weight(value: float) -> float:
    return round(float(value), 10)


def nonnegative_grid_weights(*, num_models: int, step: float = 0.05) -> list[list[float]]:
    if int(num_models) < 1:
        raise ValueError("num_models must be positive")
    units = int(round(1.0 / float(step)))
    if units <= 0 or abs(units * float(step) - 1.0) > 1e-8:
        raise ValueError("step must divide 1.0")
    if int(num_models) == 1:
        return [[1.0]]
    if int(num_models) == 2:
        return [[_round_weight(i / units), _round_weight(1.0 - i / units)] for i in range(units + 1)]

    rows: list[list[float]] = []

    def rec(prefix: list[int], remaining_units: int, remaining_models: int) -> None:
        if remaining_models == 1:
            rows.append([_round_weight(v / units) for v in prefix + [remaining_units]])
            return
        for value in range(remaining_units + 1):
            rec(prefix + [value], remaining_units - value, remaining_models - 1)

    rec([], units, int(num_models))
    return rows


def weighted_logit_ensemble(logits: Iterable[torch.Tensor], weights: Iterable[float]) -> torch.Tensor:
    tensors = [value.to(torch.float32) for value in logits]
    weight_values = [float(value) for value in weights]
    if len(tensors) != len(weight_values):
        raise ValueError("number of logits and weights must match")
    if not tensors:
        raise ValueError("at least one logit tensor is required")
    if any(weight < -1e-12 for weight in weight_values):
        raise ValueError("weights must be non-negative")
    total = sum(weight_values)
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    norm = [weight / total for weight in weight_values]
    out = torch.zeros_like(tensors[0], dtype=torch.float32)
    for tensor, weight in zip(tensors, norm):
        if tensor.shape != out.shape:
            raise ValueError("all logit tensors must have the same shape")
        out = out + float(weight) * tensor
    return out


def evaluate_ensemble_promotion(
    *,
    valid_acc: float,
    test_acc: float,
    macro_f1: float | None,
    predicted_class_count: int | None,
    best_component_valid_acc: float,
    best_component_test_acc: float,
    epsilon: float = 0.0005,
    tolerance: float = 0.001,
    component_forbidden_flags: Iterable[bool] | None = None,
    component_bounded_edges: Iterable[bool] | None = None,
) -> dict:
    reasons: list[str] = []
    if any(bool(flag) for flag in (component_forbidden_flags or [])):
        reasons.append("forbidden_component")
    if any(bool(flag) for flag in (component_bounded_edges or [])):
        reasons.append("bounded_edges_component")
    if float(valid_acc) <= float(best_component_valid_acc) + float(epsilon):
        reasons.append("validation_no_improvement")
    if float(test_acc) < float(best_component_test_acc) - float(tolerance):
        reasons.append("test_regression")

    promoted = len(reasons) == 0
    return {
        "valid_acc": float(valid_acc),
        "test_acc": float(test_acc),
        "macro_f1": "" if macro_f1 is None else float(macro_f1),
        "predicted_class_count": "" if predicted_class_count is None else int(predicted_class_count),
        "best_component_valid_acc": float(best_component_valid_acc),
        "best_component_test_acc": float(best_component_test_acc),
        "promotion_status": "promoted" if promoted else "blocked",
        "promotion_reason": "validation_and_test_gate_passed" if promoted else ",".join(reasons),
    }


def validation_softmax_weights(valid_scores: Iterable[float], temperature: float = 1.0) -> list[float]:
    scores = torch.tensor([float(value) for value in valid_scores], dtype=torch.float32)
    if scores.numel() == 0:
        return []
    weights = torch.softmax(scores / max(float(temperature), 1e-12), dim=0)
    return [float(value) for value in weights]
