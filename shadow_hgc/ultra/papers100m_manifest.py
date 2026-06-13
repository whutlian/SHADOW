from __future__ import annotations

import gzip
import time
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_contract import (
    PAPERS100M_FEATURE_DIM,
    PAPERS100M_NUM_CLASSES,
    PAPERS100M_NUM_EDGES,
    PAPERS100M_NUM_NODES,
)
from shadow_hgc.ultra.papers100m_memmap import (
    array_checksum,
    file_checksum,
    first_existing,
    git_hash,
    npy_mmap,
    read_json,
    resolve_source_root,
    stable_hash,
    utc_now,
    write_json,
    write_memmap,
)


def _load_index_array(source_root: Path, name: str, data_root: Path) -> np.ndarray:
    path = first_existing(source_root, [f"{name}.npy"])
    if path is not None:
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64).reshape(-1)
    gz = data_root / "papers100M-bin" / "split" / "time" / f"{name.replace('_idx', '')}.csv.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as handle:
            return np.loadtxt(handle, dtype=np.int64, delimiter=",").reshape(-1)
    raise FileNotFoundError(f"missing split index {name}")


def _load_meta(source_root: Path) -> dict[str, Any]:
    meta_path = first_existing(source_root, ["dataset_meta.json", "manifest.json"])
    if meta_path is None:
        return {}
    try:
        raw = read_json(meta_path)
    except Exception:
        return {}
    if "node_feat.npy" in raw:
        feat = raw.get("node_feat.npy", {})
        edge = raw.get("edge_index.npy", {})
        return {
            "dataset_name": "ogbn-papers100M",
            "num_nodes": int(feat.get("shape", [0, 0])[0]),
            "feature_dim": int(feat.get("shape", [0, 0])[1]),
            "num_edges": int(edge.get("shape", [0, 0])[1]),
            "num_classes": PAPERS100M_NUM_CLASSES,
        }
    return raw


def _validate_counts(manifest: dict[str, Any], *, allow_toy: bool) -> None:
    if allow_toy:
        return
    expected = {
        "num_nodes": PAPERS100M_NUM_NODES,
        "num_edges": PAPERS100M_NUM_EDGES,
        "feature_dim": PAPERS100M_FEATURE_DIM,
        "num_classes": PAPERS100M_NUM_CLASSES,
    }
    mismatched = [f"{key}:expected={expected[key]} actual={manifest.get(key)}" for key in expected if int(manifest.get(key, -1)) != expected[key]]
    if mismatched:
        raise ValueError("papers100M manifest validation failed: " + ", ".join(mismatched))


def _write_feature_fp16_cache(path: Path, feat: np.ndarray, *, chunk_size_nodes: int = 262_144) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(v) for v in feat.shape)
    mmap = np.memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for start in range(0, shape[0], int(chunk_size_nodes)):
        stop = min(start + int(chunk_size_nodes), shape[0])
        mmap[start:stop] = np.asarray(feat[start:stop], dtype=np.float16)
    mmap.flush()
    del mmap
    return {
        "path": path.name,
        "dtype": "float16",
        "shape": list(shape),
        "bytes": int(path.stat().st_size),
        "checksum": file_checksum(path),
    }


def build_papers100m_manifest(
    data_root: str | Path,
    cache_root: str | Path,
    *,
    allow_toy: bool = False,
    materialize_raw_features: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    source_root = resolve_source_root(data_root)
    cache_root = Path(cache_root)
    raw_dir = cache_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    feat_path = first_existing(source_root, ["node_feat.npy"])
    label_path = first_existing(source_root, ["node_label.npy", "node-label.npy"])
    edge_path = first_existing(source_root, ["edge_index.npy"])
    if feat_path is None or label_path is None or edge_path is None:
        raise FileNotFoundError(f"data root does not expose node_feat.npy, node_label.npy, and edge_index.npy: {source_root}")

    feat = npy_mmap(feat_path)
    labels = npy_mmap(label_path).reshape(-1)
    edge = npy_mmap(edge_path)
    meta = _load_meta(source_root)
    num_nodes = int(meta.get("num_nodes", feat.shape[0]))
    feature_dim = int(meta.get("feature_dim", feat.shape[1]))
    num_edges = int(meta.get("num_edges", edge.shape[1] if edge.ndim == 2 else edge.shape[0]))
    if allow_toy:
        num_classes = int(meta.get("num_classes", int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0))
    else:
        num_classes = int(meta.get("num_classes", PAPERS100M_NUM_CLASSES))

    train_idx = _load_index_array(source_root, "train_idx", data_root)
    valid_idx = _load_index_array(source_root, "valid_idx", data_root)
    test_idx = _load_index_array(source_root, "test_idx", data_root)
    target_idx = np.unique(np.concatenate([train_idx, valid_idx, test_idx]).astype(np.uint32))
    target_local_id = np.full(num_nodes, -1, dtype=np.int32)
    target_local_id[target_idx.astype(np.int64)] = np.arange(target_idx.size, dtype=np.int32)

    write_memmap(raw_dir / "target_idx.u32.memmap", target_idx, dtype=np.uint32)
    write_memmap(raw_dir / "target_local_id.i32.memmap", target_local_id, dtype=np.int32)
    write_memmap(raw_dir / "train_idx.u32.memmap", train_idx.astype(np.uint32), dtype=np.uint32)
    write_memmap(raw_dir / "valid_idx.u32.memmap", valid_idx.astype(np.uint32), dtype=np.uint32)
    write_memmap(raw_dir / "test_idx.u32.memmap", test_idx.astype(np.uint32), dtype=np.uint32)
    write_memmap(raw_dir / "train_local_idx.u32.memmap", target_local_id[train_idx].astype(np.uint32), dtype=np.uint32)
    write_memmap(raw_dir / "valid_local_idx.u32.memmap", target_local_id[valid_idx].astype(np.uint32), dtype=np.uint32)
    write_memmap(raw_dir / "test_local_idx.u32.memmap", target_local_id[test_idx].astype(np.uint32), dtype=np.uint32)
    label_values = np.asarray(labels)
    if np.issubdtype(label_values.dtype, np.floating):
        clean_labels = np.where(np.isfinite(label_values), label_values, -1)
    else:
        clean_labels = label_values
    clean_labels = np.where(clean_labels >= 0, clean_labels, -1).astype(np.int16)
    write_memmap(raw_dir / "node_label.int16.memmap", clean_labels, dtype=np.int16)

    feature_meta: dict[str, Any] = {
        "source_path": str(feat_path),
        "source_dtype": str(feat.dtype),
        "shape": [num_nodes, feature_dim],
        "cached_path": "",
        "cached_dtype": "",
    }
    if materialize_raw_features:
        feature_info = _write_feature_fp16_cache(raw_dir / "node_feat.fp16.memmap", feat)
        feature_meta.update({"cached_path": feature_info["path"], "cached_dtype": "float16", "cache_bytes": feature_info["bytes"]})
    write_json(raw_dir / "node_feat_meta.json", feature_meta)

    manifest = {
        "dataset_name": "toy-papers100M" if allow_toy and num_nodes != PAPERS100M_NUM_NODES else "ogbn-papers100M",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "train_size": int(train_idx.size),
        "valid_size": int(valid_idx.size),
        "test_size": int(test_idx.size),
        "target_universe_size": int(target_idx.size),
        "target_idx_checksum": array_checksum(target_idx),
        "split_checksum": array_checksum(np.concatenate([train_idx, valid_idx, test_idx]).astype(np.int64), max_bytes=16 * 1024 * 1024),
        "node_order_checksum": array_checksum(np.array([num_nodes, target_idx.size], dtype=np.int64)),
        "edge_checksum": array_checksum(np.array(edge.shape, dtype=np.int64)),
        "feature_checksum": array_checksum(np.array([num_nodes, feature_dim], dtype=np.int64)),
        "source_data_root": str(source_root),
        "raw_dir": "raw",
        "created_at": utc_now(),
        "code_version_or_git_hash_if_available": git_hash(),
        "manifest_build_time": float(time.perf_counter() - started),
    }
    _validate_counts(manifest, allow_toy=allow_toy)
    manifest["cache_build_id"] = stable_hash({key: manifest[key] for key in ("dataset_name", "num_nodes", "num_edges", "target_idx_checksum", "split_checksum")})
    write_json(raw_dir / "raw_manifest.json", manifest)
    write_json(cache_root / "manifest.json", manifest)
    return manifest
