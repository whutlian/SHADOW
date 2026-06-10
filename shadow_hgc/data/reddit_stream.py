from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _extract_npy_member(npz_path: Path, member: str, out_path: Path, *, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as archive:
        name = f"{member}.npy"
        with archive.open(name) as source, out_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)


def _copy_feature_float32(source_path: Path, out_path: Path, *, chunk_rows: int, overwrite: bool) -> dict[str, Any]:
    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError("Reddit feature array must have shape [num_nodes, feature_dim]")
    if out_path.exists() and not overwrite:
        existing = np.load(out_path, mmap_mode="r")
        return {"shape": [int(v) for v in existing.shape], "dtype": str(existing.dtype), "bytes": int(existing.nbytes)}
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=source.shape)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(source.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(source.shape[0]))
        out[start:stop] = source[start:stop].astype(np.float32, copy=False)
    out.flush()
    return {"shape": [int(v) for v in out.shape], "dtype": str(out.dtype), "bytes": int(out.nbytes)}


def _write_int_array(path: Path, values: np.ndarray, *, overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        existing = np.load(path, mmap_mode="r")
        return {"shape": [int(v) for v in existing.shape], "dtype": str(existing.dtype), "bytes": int(existing.nbytes)}
    np.save(path, values.astype(np.int64, copy=False))
    written = np.load(path, mmap_mode="r")
    return {"shape": [int(v) for v in written.shape], "dtype": str(written.dtype), "bytes": int(written.nbytes)}


def prepare_reddit_raw_memmaps(
    root: str | Path = "dataset/Reddit",
    *,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
    feature_chunk_rows: int = 16384,
) -> dict[str, Any]:
    """Extract Reddit raw npz files into stream-friendly .npy memmaps.

    PyG's processed `data.pt` requires a full object load. The raw Reddit npz
    files contain zipped `.npy` members, so this converter streams each member
    to disk first and keeps edges as separate one-dimensional `row`/`col`
    arrays for chunked readers.
    """

    started = time.perf_counter()
    base = Path(root)
    raw = base / "raw"
    graph_npz = raw / "reddit_graph.npz"
    data_npz = raw / "reddit_data.npz"
    if not graph_npz.exists():
        raise FileNotFoundError(f"missing Reddit graph npz: {graph_npz}")
    if not data_npz.exists():
        raise FileNotFoundError(f"missing Reddit data npz: {data_npz}")
    out = Path(out_dir) if out_dir is not None else base / "processed" / "raw_memmap"
    out.mkdir(parents=True, exist_ok=True)

    src_path = out / "src.npy"
    dst_path = out / "dst.npy"
    feature_raw_path = out / "feature.float64.npy"
    node_types_path = out / "node_types.npy"
    label_raw_path = out / "label.int32.npy"
    _extract_npy_member(graph_npz, "row", src_path, overwrite=overwrite)
    _extract_npy_member(graph_npz, "col", dst_path, overwrite=overwrite)
    _extract_npy_member(data_npz, "feature", feature_raw_path, overwrite=overwrite)
    _extract_npy_member(data_npz, "node_types", node_types_path, overwrite=overwrite)
    _extract_npy_member(data_npz, "label", label_raw_path, overwrite=overwrite)

    feature_info = _copy_feature_float32(feature_raw_path, out / "x.float32.npy", chunk_rows=feature_chunk_rows, overwrite=overwrite)
    node_types = np.load(node_types_path, mmap_mode="r")
    labels = np.load(label_raw_path, mmap_mode="r")
    label_info = _write_int_array(out / "y.int64.npy", np.asarray(labels), overwrite=overwrite)
    train_info = _write_int_array(out / "train_idx.npy", np.nonzero(np.asarray(node_types) == 1)[0], overwrite=overwrite)
    valid_info = _write_int_array(out / "valid_idx.npy", np.nonzero(np.asarray(node_types) == 2)[0], overwrite=overwrite)
    test_info = _write_int_array(out / "test_idx.npy", np.nonzero(np.asarray(node_types) == 3)[0], overwrite=overwrite)

    src = np.load(src_path, mmap_mode="r")
    dst = np.load(dst_path, mmap_mode="r")
    if src.shape != dst.shape:
        raise ValueError("Reddit raw row/col edge arrays must have the same shape")
    num_nodes = int(feature_info["shape"][0])
    num_edges = int(src.shape[0])
    num_classes = int(np.max(labels)) + 1 if labels.shape[0] else 0
    manifest = {
        "source": "reddit_raw_npz_streaming_extract",
        "memmap_root": str(out),
        "raw_graph_npz": str(graph_npz),
        "raw_data_npz": str(data_npz),
        "feature_path": "x.float32.npy",
        "feature_raw_path": "feature.float64.npy",
        "src_path": "src.npy",
        "dst_path": "dst.npy",
        "label_path": "y.int64.npy",
        "node_types_path": "node_types.npy",
        "train_idx_path": "train_idx.npy",
        "valid_idx_path": "valid_idx.npy",
        "test_idx_path": "test_idx.npy",
        "edge_orientation": "src=row,dst=col",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "feature_dim": int(feature_info["shape"][1]),
        "num_classes": num_classes,
        "train_nodes": int(train_info["shape"][0]),
        "valid_nodes": int(valid_info["shape"][0]),
        "test_nodes": int(test_info["shape"][0]),
        "feature": feature_info,
        "label": label_info,
        "train_idx": train_info,
        "valid_idx": valid_info,
        "test_idx": test_info,
        "edge_storage_bytes": int(src.nbytes + dst.nbytes),
        "converted_storage_bytes": int(
            feature_info["bytes"]
            + label_info["bytes"]
            + train_info["bytes"]
            + valid_info["bytes"]
            + test_info["bytes"]
            + src.nbytes
            + dst.nbytes
        ),
        "uses_processed_data_pt": False,
        "materialized_stacked_edge_index": False,
        "wall_time_s": float(time.perf_counter() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_reddit_raw_memmap_manifest(memmap_root: str | Path) -> dict[str, Any]:
    root = Path(memmap_root)
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def load_reddit_raw_memmap_labels_and_splits(memmap_root: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(memmap_root)
    manifest = load_reddit_raw_memmap_manifest(root)
    labels = torch.from_numpy(np.asarray(np.load(root / manifest["label_path"], mmap_mode="r"), dtype=np.int64).copy()).to(torch.long)
    train = torch.from_numpy(np.asarray(np.load(root / manifest["train_idx_path"], mmap_mode="r"), dtype=np.int64).copy()).to(torch.long)
    valid = torch.from_numpy(np.asarray(np.load(root / manifest["valid_idx_path"], mmap_mode="r"), dtype=np.int64).copy()).to(torch.long)
    test = torch.from_numpy(np.asarray(np.load(root / manifest["test_idx_path"], mmap_mode="r"), dtype=np.int64).copy()).to(torch.long)
    return labels, train, valid, test
