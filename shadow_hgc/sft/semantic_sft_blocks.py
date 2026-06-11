from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def validate_semantic_cache_alignment(
    *,
    embedding_path: str | Path,
    shape: tuple[int, int],
    num_nodes: int,
    matched_nodes: int,
    min_match_rate: float = 0.95,
) -> dict[str, Any]:
    path = Path(embedding_path)
    exists = path.exists()
    cache_bytes = path.stat().st_size if exists else 0
    shape_ok = False
    if exists:
        expected = int(shape[0]) * int(shape[1]) * np.dtype(np.float32).itemsize
        shape_ok = cache_bytes >= expected
    match_rate = float(matched_nodes) / float(max(1, num_nodes))
    unmatched = int(max(0, num_nodes - matched_nodes))
    blocked = (not exists) or (not shape_ok) or match_rate < float(min_match_rate)
    reason = ""
    if not exists:
        reason = "semantic_cache_missing"
    elif not shape_ok:
        reason = "semantic_cache_shape_mismatch"
    elif match_rate < float(min_match_rate):
        reason = "semantic_match_rate_too_low"
    return {
        "semantic_cache_path": str(path),
        "semantic_cache_bytes": int(cache_bytes),
        "semantic_feature_dim": int(shape[1]),
        "semantic_match_rate": match_rate,
        "semantic_unmatched_nodes": unmatched,
        "shape_ok": shape_ok,
        "blocked": blocked,
        "failure_reason": reason,
    }
