from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def write_semantic_cache_metadata(
    metadata_path: str | Path,
    *,
    embedding_path: str | Path,
    model_name: str,
    shape: tuple[int, int],
    dtype: str = "float16",
) -> Path:
    meta = Path(metadata_path)
    emb = Path(embedding_path)
    payload = {
        "model_name": str(model_name),
        "embedding_path": str(emb.resolve()),
        "shape": [int(shape[0]), int(shape[1])],
        "dtype": str(np.dtype(dtype)),
        "checksum": _hash_file(emb),
        "node_order_hash": hashlib.sha256(f"ogb-node-order:{shape[0]}".encode("utf-8")).hexdigest()[:16],
        "build_time": time.time(),
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def validate_semantic_cache_v3(*, metadata_path: str | Path, expected_num_nodes: int) -> dict[str, Any]:
    meta = Path(metadata_path)
    if not meta.exists():
        return {"blocked": True, "failure_reason": "semantic_cache_missing", "semantic_cache_path": str(meta)}
    payload = json.loads(meta.read_text(encoding="utf-8"))
    emb = Path(payload.get("embedding_path", ""))
    if not emb.is_absolute():
        emb = meta.parent / emb
    if not emb.exists():
        return {"blocked": True, "failure_reason": "semantic_cache_missing", "semantic_cache_path": str(emb)}
    shape = [int(v) for v in payload.get("shape", [])]
    if len(shape) != 2 or int(shape[0]) != int(expected_num_nodes):
        return {"blocked": True, "failure_reason": "semantic_shape_mismatch", "shape": shape}
    dtype = np.dtype(str(payload.get("dtype", "float16")))
    expected_bytes = int(shape[0]) * int(shape[1]) * int(dtype.itemsize)
    actual_bytes = int(emb.stat().st_size)
    if actual_bytes < expected_bytes:
        return {"blocked": True, "failure_reason": "semantic_cache_truncated", "semantic_cache_bytes": actual_bytes}
    return {
        "blocked": False,
        "failure_reason": "",
        "semantic_encoder": str(payload.get("model_name", "")),
        "semantic_cache_path": str(emb),
        "semantic_dim": int(shape[1]),
        "semantic_cache_bytes": actual_bytes,
        "semantic_cache_checksum": str(payload.get("checksum", "")),
        "node_order_hash": str(payload.get("node_order_hash", "")),
    }
