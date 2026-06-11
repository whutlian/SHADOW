from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any


HISTORICAL_LAD_MARKERS = ("lad_reference", "historical_replay", "t1_safe", "lad_cache")


def is_historical_lad_predictor(name: str | Path) -> bool:
    lowered = str(name).lower()
    return any(marker in lowered for marker in HISTORICAL_LAD_MARKERS)


def cns_grid_plan(
    *,
    correction_alphas: list[float],
    smoothing_alphas: list[float],
    correction_steps: list[int],
    smoothing_steps: list[int],
    autoscale: list[str],
    graph_directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ca, sa, cs, ss, au, direction in product(
        correction_alphas,
        smoothing_alphas,
        correction_steps,
        smoothing_steps,
        autoscale,
        graph_directions,
    ):
        rows.append(
            {
                "correction_alpha": float(ca),
                "smoothing_alpha": float(sa),
                "correction_steps": int(cs),
                "smoothing_steps": int(ss),
                "autoscale": str(au),
                "graph_direction": str(direction),
            }
        )
    return rows


def find_base_logits(base_logits_dir: str | Path, predictor: str) -> Path | None:
    root = Path(base_logits_dir)
    candidates = [
        root / f"{predictor}_logits.pt",
        root / f"{predictor}.pt",
        root / predictor / "logits.pt",
        root / predictor / "all_target_logits.memmap",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
