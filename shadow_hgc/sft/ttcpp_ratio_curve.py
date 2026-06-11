from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any

from shadow_hgc.sft.t33_contract import make_t33_row, ratio_budget


def make_ratio_curve_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    ratio: float,
    accuracy: float,
    macro_f1: float,
    valid_acc: float | str = "",
    virtual_mixup_count: int = 0,
    **fields: Any,
) -> dict[str, Any]:
    return make_t33_row(
        dataset=dataset,
        method=method,
        seed=int(seed),
        requested_full_node_ratio=float(ratio),
        total_condensed_nodes=ratio_budget(dataset, ratio),
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        valid_acc=valid_acc,
        shadow_nodes=0,
        condensed_edges=0,
        virtual_mixup_count=int(virtual_mixup_count),
        **fields,
    )


def _float(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    try:
        if row.get(field) in {"", None}:
            return default
        return float(row.get(field))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _corr(xs: list[float], ys: list[float]) -> float | str:
    if len(xs) < 3 or len(xs) != len(ys):
        return ""
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def aggregate_ratio_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("method", "")), _float(row, "requested_full_node_ratio"))].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method, ratio), group in sorted(groups.items(), key=lambda item: item[0]):
        acc = [_float(row, "accuracy") for row in group if row.get("accuracy") not in {"", None}]
        macro = [_float(row, "macro_f1") for row in group if row.get("macro_f1") not in {"", None}]
        valid = [_float(row, "valid_acc") for row in group if row.get("valid_acc") not in {"", None}]
        gaps = [abs(_float(row, "valid_acc") - _float(row, "accuracy")) for row in group if row.get("valid_acc") not in {"", None} and row.get("accuracy") not in {"", None}]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "requested_full_node_ratio": ratio,
                "seed_count": len(group),
                "accuracy_mean": _mean(acc),
                "accuracy_std": _std(acc),
                "macro_f1_mean": _mean(macro),
                "macro_f1_std": _std(macro),
                "accuracy_best": max(acc) if acc else "",
                "accuracy_median": median(acc) if acc else "",
                "accuracy_worst": min(acc) if acc else "",
                "valid_test_gap_mean": _mean(gaps),
                "valid_test_correlation": _corr(valid, acc) if valid and acc else "",
            }
        )
    return out
