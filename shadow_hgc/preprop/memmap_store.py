from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shadow_hgc.preprop.memmap_blocks import numpy_dtype, torch_dtype


def write_tensor_memmap(path: str | Path, tensor: torch.Tensor, *, dtype: str = "float16") -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = numpy_dtype(dtype)
    array = tensor.detach().cpu().to(dtype=torch_dtype(dtype)).contiguous().numpy()
    mmap = np.memmap(target, mode="w+", dtype=np_dtype, shape=array.shape)
    mmap[:] = array[:]
    mmap.flush()
    return {
        "path": str(target),
        "shape": [int(value) for value in array.shape],
        "dtype": np_dtype.name,
        "disk_bytes": int(array.size * np_dtype.itemsize),
    }


def read_tensor_memmap(path: str | Path, *, shape: list[int] | tuple[int, ...], dtype: str) -> torch.Tensor:
    array = np.memmap(Path(path), mode="r", dtype=numpy_dtype(dtype), shape=tuple(int(v) for v in shape))
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32)
