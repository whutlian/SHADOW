from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import directory_bytes, first_existing, npy_mmap, read_json, resolve_source_root, stable_hash, utc_now, write_json


def _open_source_features(cache_root: Path, root_manifest: dict[str, Any]) -> np.ndarray:
    raw_dir = cache_root / "raw"
    meta = read_json(raw_dir / "node_feat_meta.json")
    cached = raw_dir / str(meta.get("cached_path", ""))
    shape = tuple(int(v) for v in meta["shape"])
    if cached.exists() and str(meta.get("cached_dtype", "")):
        return np.memmap(cached, mode="r", dtype=np.dtype(str(meta["cached_dtype"])), shape=shape)
    source_root = resolve_source_root(root_manifest["source_data_root"])
    path = first_existing(source_root, ["node_feat.npy"])
    if path is None:
        raise FileNotFoundError("missing node_feat.npy for SFT cache")
    return npy_mmap(path)


def _target_arrays(cache_root: Path, manifest: dict[str, Any]) -> tuple[np.memmap, np.memmap]:
    target_size = int(manifest["target_universe_size"])
    num_nodes = int(manifest["num_nodes"])
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    target_local_id = np.memmap(cache_root / "raw" / "target_local_id.i32.memmap", mode="r", dtype=np.int32, shape=(num_nodes,))
    return target_idx, target_local_id


def _accumulate_x1(
    *,
    out: np.memmap,
    features: np.ndarray,
    edge_src: np.memmap,
    edge_dst: np.memmap,
    denominator: np.memmap,
    target_local_id: np.memmap,
    reverse: bool,
    chunk_size_edges: int,
) -> None:
    num_edges = int(edge_src.shape[0])
    for start in range(0, num_edges, int(chunk_size_edges)):
        stop = min(start + int(chunk_size_edges), num_edges)
        src = np.asarray(edge_src[start:stop], dtype=np.int64)
        dst = np.asarray(edge_dst[start:stop], dtype=np.int64)
        msg_src = dst if reverse else src
        msg_dst = src if reverse else dst
        local = np.asarray(target_local_id[msg_dst], dtype=np.int64)
        mask = local >= 0
        if not np.any(mask):
            continue
        denom = np.asarray(denominator[msg_dst[mask]], dtype=np.float32)
        alpha = 1.0 / np.maximum(denom, 1.0)
        vals = np.asarray(features[msg_src[mask]], dtype=np.float32) * alpha[:, None]
        np.add.at(out, local[mask], vals)


def _build_label_scalars(
    *,
    cache_root: Path,
    edge_src: np.memmap,
    edge_dst: np.memmap,
    target_local_id: np.memmap,
    dst_degree: np.memmap,
    num_classes: int,
    chunk_size_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    manifest = read_json(cache_root / "manifest.json")
    target_size = int(manifest["target_universe_size"])
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(manifest["num_nodes"]),))
    train_idx = np.memmap(cache_root / "raw" / "train_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest["train_size"]),))
    train_mask = np.zeros(int(manifest["num_nodes"]), dtype=np.bool_)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    support = np.zeros(target_size, dtype=np.float32)
    hist = np.zeros((target_size, int(num_classes)), dtype=np.uint16)
    num_edges = int(edge_src.shape[0])
    for start in range(0, num_edges, int(chunk_size_edges)):
        stop = min(start + int(chunk_size_edges), num_edges)
        src = np.asarray(edge_src[start:stop], dtype=np.int64)
        dst = np.asarray(edge_dst[start:stop], dtype=np.int64)
        local = np.asarray(target_local_id[dst], dtype=np.int64)
        mask = (local >= 0) & train_mask[src] & (labels[src] >= 0) & (labels[src] < int(num_classes))
        if not np.any(mask):
            continue
        denom = np.maximum(np.asarray(dst_degree[dst[mask]], dtype=np.float32), 1.0)
        np.add.at(support, local[mask], 1.0 / denom)
        np.add.at(hist, (local[mask], np.asarray(labels[src[mask]], dtype=np.int64)), 1)
    total = hist.sum(axis=1, keepdims=True).astype(np.float32)
    probs = np.divide(hist.astype(np.float32), np.maximum(total, 1.0), where=total >= 0)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1).astype(np.float32)
    entropy[total.reshape(-1) == 0] = 0.0
    return support.reshape(-1, 1), entropy.reshape(-1, 1)


def _write_block(path: Path, values: np.ndarray, *, dtype: str = "float16") -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.dtype(dtype))
    mmap = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mmap[:] = arr
    mmap.flush()
    del mmap
    return {"name": path.name.split(".")[0], "path": path.name, "dtype": np.dtype(dtype).name, "shape": list(arr.shape), "bytes": int(path.stat().st_size)}


def build_or_load_sft_cache(
    cache_root: str | Path,
    *,
    chunk_size_edges: int = 5_000_000,
    force: bool = False,
    x2_mode: str = "disabled",
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    sft_dir = cache_root / "sft"
    manifest_path = sft_dir / "sft_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    if str(x2_mode) != "disabled":
        raise ValueError("T35 first promoted path supports x2_mode=disabled only")
    started = time.perf_counter()
    root_manifest = read_json(cache_root / "manifest.json")
    edge_manifest = read_json(cache_root / "graph" / "edge_slice_manifest.json")
    num_nodes = int(root_manifest["num_nodes"])
    num_edges = int(root_manifest["num_edges"])
    target_size = int(root_manifest["target_universe_size"])
    feature_dim = int(root_manifest["feature_dim"])
    num_classes = int(root_manifest["num_classes"])
    sft_dir.mkdir(parents=True, exist_ok=True)

    target_idx, target_local_id = _target_arrays(cache_root, root_manifest)
    features = _open_source_features(cache_root, root_manifest)
    edge_src = np.memmap(cache_root / "graph" / "edge_src.u32.memmap", mode="r", dtype=np.uint32, shape=(num_edges,))
    edge_dst = np.memmap(cache_root / "graph" / "edge_dst.u32.memmap", mode="r", dtype=np.uint32, shape=(num_edges,))
    src_degree = np.memmap(cache_root / "graph" / "src_degree.u32.memmap", mode="r", dtype=np.uint32, shape=(num_nodes,))
    dst_degree = np.memmap(cache_root / "graph" / "dst_degree.u32.memmap", mode="r", dtype=np.uint32, shape=(num_nodes,))

    blocks: list[dict[str, Any]] = []
    x0 = np.asarray(features[np.asarray(target_idx, dtype=np.int64)], dtype=np.float16)
    blocks.append(_write_block(sft_dir / "X0_target.fp16.memmap", x0, dtype="float16"))

    x1_ref = np.memmap(sft_dir / "X1_cite_ref_target.fp32.tmp", mode="w+", dtype=np.float32, shape=(target_size, feature_dim))
    x1_ref[:] = 0.0
    _accumulate_x1(out=x1_ref, features=features, edge_src=edge_src, edge_dst=edge_dst, denominator=dst_degree, target_local_id=target_local_id, reverse=False, chunk_size_edges=chunk_size_edges)
    x1_ref.flush()
    blocks.append(_write_block(sft_dir / "X1_cite_ref_target.fp16.memmap", np.asarray(x1_ref, dtype=np.float16), dtype="float16"))
    del x1_ref
    (sft_dir / "X1_cite_ref_target.fp32.tmp").unlink(missing_ok=True)

    x1_rev = np.memmap(sft_dir / "X1_cited_by_target.fp32.tmp", mode="w+", dtype=np.float32, shape=(target_size, feature_dim))
    x1_rev[:] = 0.0
    _accumulate_x1(out=x1_rev, features=features, edge_src=edge_src, edge_dst=edge_dst, denominator=src_degree, target_local_id=target_local_id, reverse=True, chunk_size_edges=chunk_size_edges)
    x1_rev.flush()
    blocks.append(_write_block(sft_dir / "X1_cited_by_target.fp16.memmap", np.asarray(x1_rev, dtype=np.float16), dtype="float16"))
    del x1_rev
    (sft_dir / "X1_cited_by_target.fp32.tmp").unlink(missing_ok=True)

    degree_target = np.stack(
        [
            np.log1p(np.asarray(dst_degree[np.asarray(target_idx, dtype=np.int64)], dtype=np.float32)),
            np.log1p(np.asarray(src_degree[np.asarray(target_idx, dtype=np.int64)], dtype=np.float32)),
        ],
        axis=1,
    )
    blocks.append(_write_block(sft_dir / "degree_target.fp16.memmap", degree_target, dtype="float16"))
    support, entropy = _build_label_scalars(cache_root=cache_root, edge_src=edge_src, edge_dst=edge_dst, target_local_id=target_local_id, dst_degree=dst_degree, num_classes=num_classes, chunk_size_edges=chunk_size_edges)
    blocks.append(_write_block(sft_dir / "label_support_target.fp16.memmap", support, dtype="float16"))
    blocks.append(_write_block(sft_dir / "label_entropy_target.fp16.memmap", entropy, dtype="float16"))

    block_names = [block["name"] for block in blocks]
    manifest = {
        "dataset_name": root_manifest["dataset_name"],
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "target_universe_size": target_size,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "blocks": block_names,
        "block_entries": blocks,
        "x2_mode": str(x2_mode),
        "sft_cache_bytes": directory_bytes(sft_dir),
        "sft_cache_time": float(time.perf_counter() - started),
        "full_edge_scans_for_sft_cache": 3,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "created_at": utc_now(),
        "parent_cache_ids": {"cache_build_id": root_manifest.get("cache_build_id", ""), "edge_cache_id": edge_manifest.get("edge_cache_id", "")},
    }
    manifest["sft_cache_id"] = stable_hash({"blocks": block_names, "bytes": manifest["sft_cache_bytes"], "edge": edge_manifest.get("edge_cache_id", "")})
    write_json(manifest_path, manifest)
    write_json(sft_dir / "manifest.json", {"blocks": blocks})
    return manifest
