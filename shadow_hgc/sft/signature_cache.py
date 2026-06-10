from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from shadow_hgc.train.lazy_sft_memmap import LazyMemmapBlockStore, load_manifest_block_store


@dataclass(frozen=True)
class SFTSignatureCacheResult:
    root: Path
    metadata: dict[str, Any]


def _as_rows(rows: torch.Tensor | np.ndarray | list[int]) -> torch.Tensor:
    return torch.as_tensor(rows, dtype=torch.long).cpu()


def _write_memmap(path: Path, data: np.ndarray, *, dtype: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=np.dtype(dtype))
    mm = np.memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
    mm[:] = array[:]
    mm.flush()
    return {"path": str(path.name), "shape": [int(v) for v in array.shape], "dtype": str(array.dtype), "bytes": int(array.nbytes)}


def write_sft_signature_cache_from_blocks(
    *,
    blocks: Mapping[str, torch.Tensor],
    splits: Mapping[str, torch.Tensor | np.ndarray | list[int]],
    train_rows: torch.Tensor | np.ndarray | list[int],
    out_dir: str | Path,
    selected_blocks: list[str] | tuple[str, ...] | None = None,
    dtype: str = "float16",
) -> SFTSignatureCacheResult:
    names = [str(name) for name in (selected_blocks or blocks.keys())]
    train = _as_rows(train_rows)
    pieces_all: list[torch.Tensor] = []
    stats: dict[str, dict[str, Any]] = {}
    for name in names:
        block = blocks[name].to(torch.float32).cpu()
        mean = block[train].mean(dim=0, keepdim=True)
        std = block[train].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        pieces_all.append((block - mean) / std)
        stats[name] = {"dim": int(block.shape[1]), "normalization_stats_source": "train_target_rows"}
    signature = torch.cat(pieces_all, dim=1).numpy()
    root = Path(out_dir)
    arrays: dict[str, Any] = {}
    for split_name, rows in splits.items():
        idx = _as_rows(rows).numpy()
        arrays[f"{split_name}_signature"] = _write_memmap(root / f"{split_name}_signature.memmap", signature[idx], dtype=dtype)
    metadata = {
        "block_names": names,
        "block_dims": {name: stats[name]["dim"] for name in names},
        "normalization_stats_source": "train_target_rows",
        "cache_bytes": int(sum(item["bytes"] for item in arrays.values())),
        "dtype": dtype,
        "arrays": arrays,
        "uses_logits": False,
        "uses_logits_as_input": False,
        "uses_kd": False,
        "uses_teacher_logits": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return SFTSignatureCacheResult(root=root, metadata=metadata)


def write_sft_signature_cache_from_memmap(
    *,
    manifest_dir: str | Path,
    splits: Mapping[str, torch.Tensor | np.ndarray | list[int]],
    train_rows: torch.Tensor | np.ndarray | list[int],
    out_dir: str | Path,
    selected_blocks: list[str] | tuple[str, ...] | None = None,
    dtype: str = "float16",
    batch_size: int = 16_384,
) -> SFTSignatureCacheResult:
    store = load_manifest_block_store(manifest_dir).subset(selected_blocks)
    names = list(store.block_dims)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    total_bytes = 0
    dim = int(sum(store.block_dims.values()))
    for split_name, rows_in in splits.items():
        rows = _as_rows(rows_in)
        shape = (int(rows.numel()), dim)
        mm = np.memmap(root / f"{split_name}_signature.memmap", mode="w+", dtype=np.dtype(dtype), shape=shape)
        offset = 0
        for start in range(0, int(rows.numel()), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            fetched = store.fetch(batch_rows, device=torch.device("cpu"))
            signature = torch.cat([fetched[name].to(torch.float32) for name in names], dim=1).numpy()
            mm[offset : offset + signature.shape[0]] = signature.astype(dtype, copy=False)
            offset += signature.shape[0]
        mm.flush()
        bytes_used = int(np.prod(shape) * np.dtype(dtype).itemsize)
        total_bytes += bytes_used
        arrays[f"{split_name}_signature"] = {
            "path": f"{split_name}_signature.memmap",
            "shape": [int(v) for v in shape],
            "dtype": dtype,
            "bytes": bytes_used,
        }
    metadata = {
        "manifest_dir": str(manifest_dir),
        "block_names": names,
        "block_dims": dict(store.block_dims),
        "normalization_stats_source": "train_target_rows",
        "cache_bytes": int(total_bytes),
        "dtype": dtype,
        "arrays": arrays,
        "uses_logits": False,
        "uses_logits_as_input": False,
        "uses_kd": False,
        "uses_teacher_logits": False,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return SFTSignatureCacheResult(root=root, metadata=metadata)
