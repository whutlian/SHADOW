from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np


def validate_semantic_memmap(
    *,
    embedding_path: str | Path,
    shape: tuple[int, int] | list[int],
    num_nodes: int,
    dim: int,
    dtype: str = "float32",
) -> dict[str, Any]:
    path = Path(embedding_path)
    if not path.exists():
        return {"blocked": True, "failure_reason": "semantic_cache_missing", "semantic_cache_path": str(path)}
    parsed_shape = tuple(int(v) for v in shape)
    if parsed_shape != (int(num_nodes), int(dim)):
        return {
            "blocked": True,
            "failure_reason": "semantic_shape_mismatch",
            "semantic_cache_path": str(path),
            "shape": list(parsed_shape),
            "expected_shape": [int(num_nodes), int(dim)],
        }
    dtype_np = np.dtype(dtype)
    expected_bytes = int(num_nodes) * int(dim) * int(dtype_np.itemsize)
    actual_bytes = int(path.stat().st_size)
    if actual_bytes < expected_bytes:
        return {
            "blocked": True,
            "failure_reason": "semantic_cache_truncated",
            "semantic_cache_path": str(path),
            "semantic_cache_bytes": actual_bytes,
            "expected_bytes": expected_bytes,
        }
    _ = np.memmap(path, mode="r", dtype=dtype_np, shape=parsed_shape)
    return {
        "blocked": False,
        "failure_reason": "",
        "semantic_cache_path": str(path),
        "semantic_dim": int(dim),
        "semantic_cache_bytes": actual_bytes,
        "shape": [int(num_nodes), int(dim)],
        "dtype": str(dtype_np),
    }


def raw_text_map_is_readable(path: str | Path) -> bool:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return False
    suffix = "".join(target.suffixes[-2:])
    try:
        if suffix == ".gz" or target.suffix == ".gz":
            with gzip.open(target, "rt", encoding="utf-8", errors="ignore") as handle:
                handle.readline()
        else:
            with target.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.readline()
    except OSError:
        return False
    return True
