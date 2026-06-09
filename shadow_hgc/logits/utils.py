from __future__ import annotations

from typing import Any

import numpy as np
import torch


def as_numpy(value: np.ndarray | torch.Tensor | None, *, dtype: str | np.dtype | None = None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def as_torch(value: np.ndarray | torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def tensor_bytes(value: np.ndarray | torch.Tensor | None) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    array = np.asarray(value)
    return int(array.size * array.dtype.itemsize)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return float((pred.to(labels.device) == labels.to(torch.long)).to(torch.float32).mean().item())
