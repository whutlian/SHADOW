from __future__ import annotations

from typing import Any

import numpy as np


def ratio_count(ratio: float, denominator: int, limit: int) -> int:
    return min(int(limit), max(1, int(round(float(ratio) * int(denominator)))))


def audit_scr_prefixes(global_rank: np.ndarray, *, ratios: list[float], denominator: int) -> list[dict[str, Any]]:
    selected = np.asarray(global_rank, dtype=np.uint32)
    previous_positions: set[int] | None = None
    previous_ratio = ""
    rows: list[dict[str, Any]] = []
    for ratio in [float(value) for value in ratios]:
        count = ratio_count(ratio, int(denominator), int(selected.shape[0]))
        positions = set(range(count))
        overlap = "" if previous_positions is None else len(previous_positions & positions) / max(1, len(previous_positions))
        violation = 0 if previous_positions is None else len(previous_positions - positions)
        rows.append(
            {
                "ratio": ratio,
                "selected_count": count,
                "prefix_previous_ratio": previous_ratio,
                "prefix_overlap_with_previous_ratio": overlap,
                "prefix_violation_count": violation,
            }
        )
        previous_positions = positions
        previous_ratio = str(ratio)
    return rows
