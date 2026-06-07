from __future__ import annotations

from pathlib import Path

import numpy as np


def create_memmap_feature_store(path: str | Path, data: np.ndarray) -> np.memmap:
    """Create a .npy memmap feature store from an in-memory array."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(path, mode="w+", dtype=data.dtype, shape=data.shape)
    mmap[:] = data
    mmap.flush()
    return np.lib.format.open_memmap(path, mode="r", dtype=data.dtype, shape=data.shape)


def open_memmap_feature_store(path: str | Path, mode: str = "r") -> np.memmap:
    return np.load(Path(path), mmap_mode=mode)


def source_id_block_gather(
    store: np.memmap | np.ndarray,
    source_ids: np.ndarray,
    *,
    block_size: int = 8192,
) -> tuple[np.ndarray, dict[str, int]]:
    """Gather features by sorted source-id blocks, then restore request order."""

    source_ids = np.asarray(source_ids, dtype=np.int64)
    base_stats = {
        "block_size": int(block_size),
        "num_unique_source_ids": int(np.unique(source_ids).size),
        "random_access_avoidance_enabled": True,
    }
    if source_ids.size == 0:
        return np.empty((0, store.shape[1]), dtype=store.dtype), {"num_blocks": 0, **base_stats}

    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    gathered_sorted = np.empty((len(source_ids), store.shape[1]), dtype=store.dtype)
    num_blocks = 0
    start = 0
    while start < len(sorted_ids):
        block_start_id = int(sorted_ids[start])
        block_end_id = block_start_id + block_size
        stop = start
        while stop < len(sorted_ids) and int(sorted_ids[stop]) < block_end_id:
            stop += 1
        gathered_sorted[start:stop] = store[sorted_ids[start:stop]]
        num_blocks += 1
        start = stop

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return gathered_sorted[inverse], {"num_blocks": int(num_blocks), **base_stats}
