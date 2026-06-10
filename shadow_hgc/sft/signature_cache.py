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


def _signature_block_key(name: str) -> str:
    return "self" if str(name) == "X0" else str(name).lower()


def load_existing_sft_signature_cache(
    out_dir: str | Path,
    *,
    manifest_dir: str | Path | None = None,
    selected_blocks: list[str] | tuple[str, ...] | None = None,
    train_rows: torch.Tensor | np.ndarray | list[int] | None = None,
    dtype: str | None = None,
) -> SFTSignatureCacheResult | None:
    root = Path(out_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = metadata.get("arrays", {})
    if not arrays:
        return None
    for array in arrays.values():
        path = root / str(array.get("path", ""))
        if not path.exists():
            return None
        expected = int(array.get("bytes", 0))
        if expected > 0 and path.stat().st_size < expected:
            return None
    if manifest_dir is not None:
        stored_manifest = metadata.get("manifest_dir")
        if stored_manifest in {"", None}:
            return None
        if Path(str(stored_manifest)).resolve() != Path(manifest_dir).resolve():
            return None
    if selected_blocks is not None:
        stored_blocks = metadata.get("block_names")
        if not stored_blocks:
            return None
        requested = [_signature_block_key(str(name)) for name in selected_blocks]
        if list(stored_blocks) != requested:
            return None
    if train_rows is not None:
        train_meta = arrays.get("train_signature")
        if train_meta is None:
            return None
        if int(train_meta.get("shape", [0])[0]) != int(_as_rows(train_rows).numel()):
            return None
    if dtype is not None and str(metadata.get("dtype")) != str(dtype):
        return None
    return SFTSignatureCacheResult(root=root, metadata=metadata)


def write_or_load_sft_signature_cache_from_memmap(
    *,
    manifest_dir: str | Path,
    splits: Mapping[str, torch.Tensor | np.ndarray | list[int]],
    train_rows: torch.Tensor | np.ndarray | list[int],
    out_dir: str | Path,
    selected_blocks: list[str] | tuple[str, ...] | None = None,
    dtype: str = "float16",
    batch_size: int = 16_384,
    reuse_existing: bool = True,
) -> SFTSignatureCacheResult:
    if reuse_existing:
        existing = load_existing_sft_signature_cache(
            out_dir,
            manifest_dir=manifest_dir,
            selected_blocks=selected_blocks,
            train_rows=train_rows,
            dtype=dtype,
        )
        if existing is not None:
            return existing
    return write_sft_signature_cache_from_memmap(
        manifest_dir=manifest_dir,
        splits=splits,
        train_rows=train_rows,
        out_dir=out_dir,
        selected_blocks=selected_blocks,
        dtype=dtype,
        batch_size=batch_size,
    )
