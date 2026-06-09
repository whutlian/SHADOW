from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def write_target_table_memmap(path: str | Path, tensor: torch.Tensor, metadata: dict | None = None) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().contiguous().numpy()
    data_path = root / "table.npy"
    np.save(data_path, array)
    meta = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
        **(metadata or {}),
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return root


def read_target_table_memmap(path: str | Path, *, mmap_mode: str | None = "r") -> tuple[np.ndarray, dict]:
    root = Path(path)
    table = np.load(root / "table.npy", mmap_mode=mmap_mode)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return table, meta
