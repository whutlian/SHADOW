from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def products_floor_for_ratio(ratio: float, *, total_budget: int, num_classes: int) -> int:
    requested = 1
    if float(ratio) >= 0.005:
        requested = 128
    elif float(ratio) >= 0.0025:
        requested = 64
    elif float(ratio) >= 0.001:
        requested = 32
    elif float(ratio) >= 0.0005:
        requested = 16
    if int(num_classes) <= 0:
        return 0
    affordable = max(1, int(total_budget) // int(num_classes))
    return max(1, min(int(requested), affordable))


def _allocate_largest_remainder(weights: dict[int, float], total: int) -> dict[int, int]:
    classes = sorted(weights)
    raw = {cls: float(weights[cls]) * int(total) for cls in classes}
    out = {cls: int(torch.floor(torch.tensor(raw[cls])).item()) for cls in classes}
    remaining = int(total) - sum(out.values())
    order = sorted(classes, key=lambda cls: (raw[cls] - out[cls], -cls), reverse=True)
    for cls in order[:remaining]:
        out[cls] += 1
    return out


def mixed_class_budget(
    labels: torch.Tensor,
    rows: torch.Tensor,
    *,
    total_budget: int,
    ratio: float,
    num_classes: int,
    seed: int = 42,
) -> dict[int, int]:
    del seed
    labels = labels.to(torch.long).cpu()
    rows = rows.to(torch.long).cpu()
    classes = list(range(int(num_classes)))
    if not classes:
        return {}
    total_budget = max(len(classes), int(total_budget))
    y = labels[rows]
    counts = {cls: int((y == cls).sum().item()) for cls in classes}
    floor = products_floor_for_ratio(float(ratio), total_budget=total_budget, num_classes=len(classes))
    budget = {cls: min(floor, max(1, counts[cls])) if counts[cls] > 0 else 0 for cls in classes}
    while sum(budget.values()) > total_budget and any(value > 1 for value in budget.values()):
        cls = max((c for c in classes if budget[c] > 1), key=lambda c: (budget[c], counts[c]))
        budget[cls] -= 1
    remaining = total_budget - sum(budget.values())
    if remaining <= 0:
        return budget
    count_sum = sum(counts.values()) or 1
    sqrt_sum = sum(counts[cls] ** 0.5 for cls in classes) or 1.0
    uniform = 1.0 / max(1, len(classes))
    weights = {
        cls: 0.45 * (counts[cls] / count_sum) + 0.35 * ((counts[cls] ** 0.5) / sqrt_sum) + 0.20 * uniform
        for cls in classes
        if counts[cls] > 0
    }
    add = _allocate_largest_remainder(weights, remaining)
    for cls, value in add.items():
        budget[cls] += int(value)
    while sum(budget.values()) < total_budget:
        cls = max((c for c in classes if counts[c] > 0), key=lambda c: counts[c] / max(1, budget[c]))
        budget[cls] += 1
    return budget


def _pairwise_distance(a: torch.Tensor, b: torch.Tensor, *, metric: str) -> torch.Tensor:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if metric == "cosine":
        a_norm = torch.nn.functional.normalize(a, dim=1)
        b_norm = torch.nn.functional.normalize(b, dim=1)
        return 1.0 - a_norm @ b_norm.t()
    if metric != "euclidean":
        raise ValueError(f"unsupported oracle metric: {metric}")
    return torch.cdist(a, b, p=2)


def nearest_prototype_oracle(
    train_signature: torch.Tensor,
    train_labels: torch.Tensor,
    selected_pos: torch.Tensor,
    eval_signature: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    metric: str = "euclidean",
) -> dict[str, Any]:
    train_signature = train_signature.to(torch.float32).cpu()
    train_labels = train_labels.to(torch.long).cpu()
    selected_pos = selected_pos.to(torch.long).cpu()
    eval_signature = eval_signature.to(torch.float32).cpu()
    eval_labels = eval_labels.to(torch.long).cpu()
    if selected_pos.numel() == 0 or eval_signature.numel() == 0:
        return {"prototype_oracle_acc": "", "centroid_oracle_acc": ""}
    proto_sig = train_signature[selected_pos]
    proto_labels = train_labels[selected_pos]
    nearest = torch.argmin(_pairwise_distance(eval_signature, proto_sig, metric=metric), dim=1)
    proto_pred = proto_labels[nearest]
    classes = torch.unique(train_labels, sorted=True)
    centroids = torch.stack([train_signature[train_labels == cls].mean(dim=0) for cls in classes], dim=0)
    centroid_nearest = torch.argmin(_pairwise_distance(eval_signature, centroids, metric=metric), dim=1)
    centroid_pred = classes[centroid_nearest]
    return {
        "prototype_oracle_acc": float((proto_pred == eval_labels).to(torch.float32).mean().item()),
        "centroid_oracle_acc": float((centroid_pred == eval_labels).to(torch.float32).mean().item()),
        "prototype_oracle_metric": metric,
    }


def per_class_collapse_report(
    labels: torch.Tensor,
    selected_rows: torch.Tensor,
    predicted_labels: torch.Tensor | None,
    *,
    num_classes: int,
    budget: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    labels = labels.to(torch.long).cpu()
    selected_rows = selected_rows.to(torch.long).cpu()
    selected_labels = labels[selected_rows] if selected_rows.numel() else labels[:0]
    predicted = None if predicted_labels is None else predicted_labels.to(torch.long).cpu()
    rows: list[dict[str, Any]] = []
    for cls in range(int(num_classes)):
        train_count = int((labels == cls).sum().item())
        selected_count = int((selected_labels == cls).sum().item())
        predicted_count = "" if predicted is None else int((predicted == cls).sum().item())
        collapsed = selected_count == 0 or (predicted is not None and int(predicted_count) == 0)
        rows.append(
            {
                "class_id": int(cls),
                "train_count": train_count,
                "budget": "" if budget is None else int(budget.get(cls, 0)),
                "selected_count": selected_count,
                "predicted_count": predicted_count,
                "collapsed": bool(collapsed),
            }
        )
    return rows


def compute_p0_recovery_diagnostics(
    *,
    alltrain_acc: float | str,
    self_fit_acc: float | str,
    normalization_match: bool,
    predicted_class_count: int | str,
    num_classes: int,
) -> dict[str, Any]:
    alltrain = None if alltrain_acc == "" else float(alltrain_acc)
    self_fit = None if self_fit_acc == "" else float(self_fit_acc)
    predicted = 0 if predicted_class_count == "" else int(predicted_class_count)
    pred_floor = min(int(num_classes), 45)
    return {
        "p0a_alltrain_acc": "" if alltrain is None else alltrain,
        "p0a_passed": False if alltrain is None else bool(alltrain >= 0.74),
        "p0b_self_fit_acc": "" if self_fit is None else self_fit,
        "p0b_passed": False if self_fit is None else bool(self_fit >= 0.95),
        "p0e_predicted_class_collapse": bool(predicted < pred_floor),
        "p0f_normalization_parity": bool(normalization_match),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def budget_to_json(budget: dict[int, int]) -> str:
    return json.dumps({str(key): int(value) for key, value in sorted(budget.items())}, sort_keys=True)
