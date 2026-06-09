from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def numpy_dtype(dtype: str) -> np.dtype:
    if dtype in {"float16", "fp16"}:
        return np.dtype("float16")
    if dtype in {"float32", "fp32"}:
        return np.dtype("float32")
    raise ValueError("dtype must be float16 or float32")


def torch_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if numpy_dtype(dtype) == np.dtype("float16") else torch.float32


def write_memmap_block(path: str | Path, block: torch.Tensor, *, dtype: str) -> dict:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = numpy_dtype(dtype)
    array = block.detach().cpu().to(dtype=torch_dtype(dtype)).contiguous().numpy()
    mmap = np.memmap(out_path, mode="w+", dtype=np_dtype, shape=array.shape)
    mmap[:] = array[:]
    mmap.flush()
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": np_dtype.name,
        "cache_bytes": int(array.size * np_dtype.itemsize),
    }
