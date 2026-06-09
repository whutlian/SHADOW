from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.logits.metadata import LogitCacheMeta, forbidden_reasons
from shadow_hgc.logits.utils import as_numpy


@dataclass(frozen=True)
class LoadedLogitCache:
    cache_dir: Path
    meta: LogitCacheMeta
    train_logits: np.ndarray
    valid_logits: np.ndarray
    test_logits: np.ndarray
    all_target_logits: np.ndarray | np.memmap | None
    y_train: np.ndarray
    y_valid: np.ndarray | None
    y_test: np.ndarray | None
    train_idx: np.ndarray
    valid_idx: np.ndarray | None
    test_idx: np.ndarray | None
    storage: dict[str, Any]


def _save_optional_npz(path: Path, arrays: dict[str, np.ndarray | None]) -> dict[str, bool]:
    present: dict[str, bool] = {}
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        present[key] = value is not None
        payload[key] = np.asarray([] if value is None else value)
    np.savez(path, **payload)
    return present


def _load_optional(npz: np.lib.npyio.NpzFile, key: str, present: dict[str, bool]) -> np.ndarray | None:
    if not bool(present.get(key, False)):
        return None
    return np.asarray(npz[key])


def _write_memmap(path: Path, array: np.ndarray, dtype: np.dtype) -> dict[str, Any]:
    mem = np.memmap(path, dtype=dtype, mode="w+", shape=array.shape)
    mem[:] = array.astype(dtype, copy=False)
    mem.flush()
    return {
        "file": path.name,
        "dtype": str(np.dtype(dtype)),
        "shape": list(array.shape),
        "bytes": int(path.stat().st_size),
        "storage": "memmap",
    }


def save_logits_cache(
    out_dir: str | Path,
    *,
    train_logits: np.ndarray | torch.Tensor,
    valid_logits: np.ndarray | torch.Tensor,
    test_logits: np.ndarray | torch.Tensor,
    all_target_logits: np.ndarray | torch.Tensor | None,
    y_train: np.ndarray | torch.Tensor,
    y_valid: np.ndarray | torch.Tensor | None,
    y_test: np.ndarray | torch.Tensor | None,
    train_idx: np.ndarray | torch.Tensor,
    valid_idx: np.ndarray | torch.Tensor | None,
    test_idx: np.ndarray | torch.Tensor | None,
    meta: LogitCacheMeta,
    dtype: str = "float16",
) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    logit_dtype = np.dtype(dtype)

    train_array = as_numpy(train_logits, dtype=logit_dtype)
    valid_array = as_numpy(valid_logits, dtype=logit_dtype)
    test_array = as_numpy(test_logits, dtype=logit_dtype)
    if train_array is None or valid_array is None or test_array is None:
        raise ValueError("train_logits, valid_logits, and test_logits are required")
    if train_array.ndim != 2 or valid_array.ndim != 2 or test_array.ndim != 2:
        raise ValueError("split logits must have shape [num_rows, num_classes]")
    if train_array.shape[1] != meta.num_classes or valid_array.shape[1] != meta.num_classes or test_array.shape[1] != meta.num_classes:
        raise ValueError("logit class dimension does not match metadata")

    np.save(root / "train_logits.npy", train_array)
    np.save(root / "valid_logits.npy", valid_array)
    np.save(root / "test_logits.npy", test_array)

    labels_present = _save_optional_npz(
        root / "labels.npz",
        {
            "y_train": as_numpy(y_train, dtype=np.int64),
            "y_valid": as_numpy(y_valid, dtype=np.int64),
            "y_test": as_numpy(y_test, dtype=np.int64),
        },
    )
    indices_present = _save_optional_npz(
        root / "indices.npz",
        {
            "train_idx": as_numpy(train_idx, dtype=np.int64),
            "valid_idx": as_numpy(valid_idx, dtype=np.int64),
            "test_idx": as_numpy(test_idx, dtype=np.int64),
        },
    )

    storage: dict[str, Any] = {
        "version": 1,
        "dtype": str(logit_dtype),
        "train_logits": {"file": "train_logits.npy", "shape": list(train_array.shape), "bytes": int((root / "train_logits.npy").stat().st_size)},
        "valid_logits": {"file": "valid_logits.npy", "shape": list(valid_array.shape), "bytes": int((root / "valid_logits.npy").stat().st_size)},
        "test_logits": {"file": "test_logits.npy", "shape": list(test_array.shape), "bytes": int((root / "test_logits.npy").stat().st_size)},
        "labels_present": labels_present,
        "indices_present": indices_present,
        "forbidden_reasons": forbidden_reasons(meta),
    }
    all_array = as_numpy(all_target_logits, dtype=logit_dtype)
    if all_array is not None:
        if all_array.ndim != 2 or all_array.shape != (meta.num_target_nodes, meta.num_classes):
            raise ValueError("all_target_logits must match [num_target_nodes, num_classes]")
        storage["all_target_logits"] = _write_memmap(root / "all_target_logits.memmap", all_array, logit_dtype)
    else:
        storage["all_target_logits"] = None

    payload = {"meta": meta.to_dict(), "storage": storage}
    (root / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def load_logits_cache(cache_dir: str | Path) -> LoadedLogitCache:
    root = Path(cache_dir)
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    meta = LogitCacheMeta.from_dict(payload["meta"])
    storage = payload.get("storage", {})

    train_logits = np.load(root / "train_logits.npy", mmap_mode="r")
    valid_logits = np.load(root / "valid_logits.npy", mmap_mode="r")
    test_logits = np.load(root / "test_logits.npy", mmap_mode="r")

    all_spec = storage.get("all_target_logits")
    all_target_logits: np.ndarray | np.memmap | None = None
    if all_spec:
        all_target_logits = np.memmap(
            root / all_spec["file"],
            dtype=np.dtype(all_spec["dtype"]),
            mode="r",
            shape=tuple(all_spec["shape"]),
        )

    labels = np.load(root / "labels.npz")
    indices = np.load(root / "indices.npz")
    labels_present = storage.get("labels_present", {})
    indices_present = storage.get("indices_present", {})
    return LoadedLogitCache(
        cache_dir=root,
        meta=meta,
        train_logits=np.asarray(train_logits),
        valid_logits=np.asarray(valid_logits),
        test_logits=np.asarray(test_logits),
        all_target_logits=all_target_logits,
        y_train=np.asarray(labels["y_train"]),
        y_valid=_load_optional(labels, "y_valid", labels_present),
        y_test=_load_optional(labels, "y_test", labels_present),
        train_idx=np.asarray(indices["train_idx"]),
        valid_idx=_load_optional(indices, "valid_idx", indices_present),
        test_idx=_load_optional(indices, "test_idx", indices_present),
        storage=storage,
    )
