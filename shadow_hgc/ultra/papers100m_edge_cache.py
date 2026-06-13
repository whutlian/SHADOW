from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import (
    array_checksum,
    directory_bytes,
    first_existing,
    npy_mmap,
    read_json,
    resolve_source_root,
    stable_hash,
    utc_now,
    write_json,
)


def _edge_chunks(edge_index: np.ndarray, chunk_size: int) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    num_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else int(edge_index.shape[0])
    for start in range(0, num_edges, int(chunk_size)):
        stop = min(start + int(chunk_size), num_edges)
        if edge_index.ndim == 2:
            src = np.asarray(edge_index[0, start:stop], dtype=np.uint32)
            dst = np.asarray(edge_index[1, start:stop], dtype=np.uint32)
        else:
            src = np.asarray(edge_index[start:stop, 0], dtype=np.uint32)
            dst = np.asarray(edge_index[start:stop, 1], dtype=np.uint32)
        yield start, stop, src, dst


def build_or_load_edge_slice_cache(
    cache_root: str | Path,
    *,
    data_root: str | Path | None = None,
    chunk_size_edges: int = 5_000_000,
    force: bool = False,
) -> dict:
    cache_root = Path(cache_root)
    graph_dir = cache_root / "graph"
    manifest_path = graph_dir / "edge_slice_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)

    root_manifest = read_json(cache_root / "manifest.json")
    source_root = resolve_source_root(data_root or root_manifest["source_data_root"])
    edge_path = first_existing(source_root, ["edge_index.npy"])
    if edge_path is None:
        raise FileNotFoundError(f"missing edge_index.npy under {source_root}")
    edge_index = npy_mmap(edge_path)
    num_nodes = int(root_manifest["num_nodes"])
    num_edges = int(root_manifest["num_edges"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    edge_src = np.memmap(graph_dir / "edge_src.u32.memmap", mode="w+", dtype=np.uint32, shape=(num_edges,))
    edge_dst = np.memmap(graph_dir / "edge_dst.u32.memmap", mode="w+", dtype=np.uint32, shape=(num_edges,))
    src_degree = np.zeros(num_nodes, dtype=np.uint32)
    dst_degree = np.zeros(num_nodes, dtype=np.uint32)
    chunks = 0
    for start, stop, src, dst in _edge_chunks(edge_index, int(chunk_size_edges)):
        edge_src[start:stop] = src
        edge_dst[start:stop] = dst
        np.add.at(src_degree, src.astype(np.int64), 1)
        np.add.at(dst_degree, dst.astype(np.int64), 1)
        chunks += 1
    edge_src.flush()
    edge_dst.flush()
    del edge_src, edge_dst

    src_map = np.memmap(graph_dir / "src_degree.u32.memmap", mode="w+", dtype=np.uint32, shape=(num_nodes,))
    dst_map = np.memmap(graph_dir / "dst_degree.u32.memmap", mode="w+", dtype=np.uint32, shape=(num_nodes,))
    src_map[:] = src_degree
    dst_map[:] = dst_degree
    src_map.flush()
    dst_map.flush()
    del src_map, dst_map
    build_time = time.perf_counter() - started

    manifest = {
        "dataset_name": root_manifest["dataset_name"],
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "target_universe_size": int(root_manifest["target_universe_size"]),
        "num_classes": int(root_manifest["num_classes"]),
        "edge_cache_bytes": directory_bytes(graph_dir),
        "edge_build_time": float(build_time),
        "edge_chunks": int(chunks),
        "chunk_size_edges": int(chunk_size_edges),
        "src_degree_checksum": array_checksum(src_degree, max_bytes=32 * 1024 * 1024),
        "dst_degree_checksum": array_checksum(dst_degree, max_bytes=32 * 1024 * 1024),
        "full_edge_scans_for_edge_cache": 1,
        "uses_full_edge_index_on_gpu": False,
        "uses_dense_p2": False,
        "uses_e_by_d_materialization": False,
        "created_at": utc_now(),
        "parent_cache_ids": {"cache_build_id": root_manifest.get("cache_build_id", "")},
    }
    manifest["edge_cache_id"] = stable_hash(
        {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "src_degree_checksum": manifest["src_degree_checksum"],
            "dst_degree_checksum": manifest["dst_degree_checksum"],
        }
    )
    manifest["edge_slice_cache_id"] = manifest["edge_cache_id"]
    write_json(graph_dir / "edge_chunk_manifest.json", manifest)
    write_json(manifest_path, manifest)
    return manifest
