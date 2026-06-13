from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DTYPES = {
    "fp16": np.float16,
    "float16": np.float16,
    "fp32": np.float32,
    "float32": np.float32,
}


def _read_checksum(value: str | Path) -> str:
    if str(value).strip() == "":
        return ""
    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return str(value).strip()


def validate_precomputed_semantic_memmap(
    *,
    memmap_path: str | Path,
    semantic_node_order_checksum: str | Path,
    expected_node_order_checksum: str,
    num_nodes: int,
    semantic_dim: int,
    semantic_dtype: str = "fp16",
) -> dict[str, Any]:
    path = Path(memmap_path)
    dtype = DTYPES.get(str(semantic_dtype).lower())
    if dtype is None:
        return {"blocked": True, "failure_reason": "unsupported_semantic_dtype", "semantic_cache_path": str(path)}
    if not path.exists():
        return {"blocked": True, "failure_reason": "raw_text_or_semantic_cache_missing", "semantic_cache_path": str(path)}
    actual_checksum = _read_checksum(semantic_node_order_checksum)
    if actual_checksum != str(expected_node_order_checksum).strip():
        return {
            "blocked": True,
            "failure_reason": "semantic_node_order_checksum_mismatch",
            "semantic_cache_path": str(path),
            "semantic_node_order_checksum": actual_checksum,
        }
    expected_bytes = int(num_nodes) * int(semantic_dim) * np.dtype(dtype).itemsize
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        return {
            "blocked": True,
            "failure_reason": "semantic_memmap_shape_mismatch",
            "semantic_cache_path": str(path),
            "semantic_cache_bytes": actual_bytes,
            "expected_semantic_cache_bytes": expected_bytes,
        }
    return {
        "blocked": False,
        "failure_reason": "",
        "semantic_cache_path": str(path),
        "semantic_cache_bytes": actual_bytes,
        "semantic_dim": int(semantic_dim),
        "semantic_dtype": str(semantic_dtype),
        "semantic_cache_memmap": True,
        "semantic_features_are_frozen": True,
        "lm_finetuned": False,
        "semantic_node_order_checksum": actual_checksum,
    }
