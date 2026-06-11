from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, 0.0)
    if value in {"", None}:
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_products_maintenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        groups[(str(row.get("method", "")), _float(row, "requested_full_node_ratio"))].append(row)
    out: list[dict[str, Any]] = []
    for (method, ratio), group in sorted(groups.items(), key=lambda item: item[0]):
        acc = [_float(row, "accuracy") for row in group]
        macro = [_float(row, "macro_f1") for row in group]
        pred = [int(float(row.get("predicted_classes", 0) or 0)) for row in group]
        out.append(
            {
                "method": method,
                "requested_full_node_ratio": ratio,
                "seed_count": len(group),
                "accuracy_mean": _mean(acc),
                "accuracy_std": _std(acc),
                "macro_f1_mean": _mean(macro),
                "macro_f1_std": _std(macro),
                "predicted_classes_mean": _mean([float(v) for v in pred]),
                "predicted_classes_min": min(pred) if pred else 0,
                "predicted_classes_max": max(pred) if pred else 0,
            }
        )
    return out
