from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def stable_hash(payload: Any) -> str:
    text = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def array_checksum(values: np.ndarray, *, max_bytes: int | None = None) -> str:
    arr = np.asarray(values)
    h = hashlib.sha256()
    if max_bytes is None or arr.nbytes <= max_bytes:
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    else:
        flat = np.ascontiguousarray(arr.reshape(-1))
        item_bytes = max(1, flat.dtype.itemsize)
        take = max(1, int(max_bytes // item_bytes))
        head = flat[: take // 2]
        tail = flat[-max(1, take - head.size) :]
        h.update(np.ascontiguousarray(head).view(np.uint8))
        h.update(np.ascontiguousarray(tail).view(np.uint8))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
    return h.hexdigest()[:16]


def file_checksum(path: str | Path, *, sample_bytes: int = 8 * 1024 * 1024) -> str:
    target = Path(path)
    h = hashlib.sha256()
    size = target.stat().st_size if target.exists() else 0
    h.update(str(size).encode("utf-8"))
    if size:
        with target.open("rb") as handle:
            h.update(handle.read(sample_bytes // 2))
            if size > sample_bytes:
                handle.seek(max(0, size - sample_bytes // 2))
                h.update(handle.read(sample_bytes // 2))
    return h.hexdigest()[:16]


def write_memmap(path: str | Path, values: np.ndarray, *, dtype: str | np.dtype) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    array = np.asarray(values, dtype=dtype)
    mmap = np.memmap(target, mode="w+", dtype=dtype, shape=array.shape)
    mmap[:] = array
    mmap.flush()
    del mmap
    return {
        "path": target.name,
        "dtype": dtype.name,
        "shape": list(array.shape),
        "bytes": int(target.stat().st_size),
        "checksum": file_checksum(target),
    }


def open_memmap(path: str | Path, *, dtype: str | np.dtype, shape: tuple[int, ...] | list[int], mode: str = "r") -> np.memmap:
    return np.memmap(Path(path), mode=mode, dtype=np.dtype(dtype), shape=tuple(int(v) for v in shape))


def directory_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return int(sum(item.stat().st_size for item in root.rglob("*") if item.is_file()))


def npy_mmap(path: str | Path) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")


def resolve_source_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    nested = root / "processed" / "papers100m_memmap"
    if nested.exists():
        return nested
    nested_bin = root / "papers100M-bin" / "processed" / "papers100m_memmap"
    if nested_bin.exists():
        return nested_bin
    return root


def first_existing(root: str | Path, names: list[str]) -> Path | None:
    base = Path(root)
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None
