from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json


def _ratio_dir_name(ratio: float) -> str:
    return f"ratio={float(ratio):.12g}".replace("+", "")


def train_or_load_ant_link_predictor(
    cache_root: str | Path,
    *,
    nested_bank_id: str,
    teacher_id: str,
    seed: int = 7,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(cache_root) / "ant" / f"bank={nested_bank_id}_teacher={teacher_id}_seed{int(seed)}"
    manifest_path = root / "link_predictor_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "papers100m_ant_link_predictor_v1",
        "nested_bank_id": str(nested_bank_id),
        "teacher_id": str(teacher_id),
        "seed": int(seed),
        "trained_once": True,
        "uses_exact_all_pair_distance": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "train_time": float(time.perf_counter() - started),
        "created_at": utc_now(),
    }
    manifest["ant_link_predictor_id"] = stable_hash({"kind": "ant_v1", "bank": nested_bank_id, "teacher": teacher_id, "seed": int(seed)})
    write_json(manifest_path, manifest)
    return manifest


def _bounded_candidates_by_class(pred_class: np.ndarray, bucket: np.ndarray, *, edge_topk: int, candidate_multiplier: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(pred_class.shape[0])
    cap = max(int(edge_topk), 1)
    candidate_cap = max(cap, int(edge_topk) * max(1, int(candidate_multiplier)))
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    total_candidates = 0
    order = np.lexsort((np.arange(n, dtype=np.int64), bucket.astype(np.int64), pred_class.astype(np.int64)))
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)
    keys = pred_class.astype(np.int64) * 1024 + bucket.astype(np.int64)
    ordered_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1]
    stops = np.r_[starts[1:], n]
    key_to_span = {int(ordered_keys[start]): (int(start), int(stop)) for start, stop in zip(starts, stops)}
    for dst in range(n):
        key = int(keys[dst])
        start, stop = key_to_span[key]
        span = stop - start
        if span <= 1:
            continue
        pos = int(inverse[dst])
        offsets = np.arange(1, min(candidate_cap + 1, span), dtype=np.int64)
        candidate_pos = start + ((pos - start + offsets) % span)
        candidates = order[candidate_pos]
        candidates = candidates[candidates != dst][:cap]
        if candidates.size == 0:
            continue
        src_chunks.append(candidates.astype(np.uint32))
        dst_chunks.append(np.full(candidates.size, dst, dtype=np.uint32))
        total_candidates += int(min(candidate_cap, max(0, span - 1)))
    if not src_chunks:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32), total_candidates
    return np.concatenate(src_chunks), np.concatenate(dst_chunks), total_candidates


def materialize_ant_edges(
    cache_root: str | Path,
    *,
    policy: str,
    seed: int,
    ratio: float,
    edge_topk: int,
    link_predictor_id: str,
    candidate_multiplier: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    bank_root = cache_root / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    bank_manifest = read_json(bank_root / "bank_manifest.json")
    max_rows = int(bank_manifest["selected_max_rows"])
    denominator = int(bank_manifest["full_node_ratio_denominator"])
    count = min(max_rows, max(1, int(round(float(ratio) * denominator))))
    out_dir = cache_root / "condensed" / _ratio_dir_name(float(ratio)) / f"ant_edges_topk{int(edge_topk)}"
    manifest_path = out_dir / "ant_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(np.memmap(bank_root / "pred_class.u16.memmap", mode="r", dtype=np.uint16, shape=(max_rows,))[:count], dtype=np.uint16)
    if (bank_root / "bucket_id.u16.memmap").exists():
        bucket = np.asarray(np.memmap(bank_root / "bucket_id.u16.memmap", mode="r", dtype=np.uint16, shape=(max_rows,))[:count], dtype=np.uint16)
    else:
        bucket = np.zeros(count, dtype=np.uint16)
    if (bank_root / "confidence.f16.memmap").exists():
        confidence = np.asarray(np.memmap(bank_root / "confidence.f16.memmap", mode="r", dtype=np.float16, shape=(max_rows,))[:count], dtype=np.float32)
    else:
        confidence = np.ones(count, dtype=np.float32)
    edge_src, edge_dst, candidate_count = _bounded_candidates_by_class(pred, bucket, edge_topk=int(edge_topk), candidate_multiplier=int(candidate_multiplier))
    if edge_src.size:
        weight = np.sqrt(np.maximum(confidence[edge_src.astype(np.int64)] * confidence[edge_dst.astype(np.int64)], 0.0)).astype(np.float16)
    else:
        weight = np.empty(0, dtype=np.float16)
    for name, values, dtype in (
        ("edge_src.u32.memmap", edge_src, np.uint32),
        ("edge_dst.u32.memmap", edge_dst, np.uint32),
        ("edge_weight.fp16.memmap", weight, np.float16),
    ):
        mm = np.memmap(out_dir / name, mode="w+", dtype=dtype, shape=values.shape)
        mm[:] = values.astype(dtype, copy=False)
        mm.flush()
        del mm
    manifest = {
        "method": f"papers100m_ant_edge_translate_topk{int(edge_topk)}",
        "ratio": float(ratio),
        "edge_topk": int(edge_topk),
        "condensed_nodes": int(count),
        "ant_edges": int(edge_src.size),
        "ant_candidate_count": int(candidate_count),
        "candidate_bound": int(count * int(edge_topk) * max(1, int(candidate_multiplier))),
        "ant_bounded": int(candidate_count) <= int(count * int(edge_topk) * max(1, int(candidate_multiplier))),
        "ant_link_predictor_id": str(link_predictor_id),
        "edge_weight_nonnegative": bool(np.all(weight >= 0)) if weight.size else True,
        "uses_exact_all_pair_distance": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "materialize_time": float(time.perf_counter() - started),
        "ant_bytes": directory_bytes(out_dir),
        "created_at": utc_now(),
    }
    manifest["ant_edge_cache_id"] = stable_hash(
        {"ratio": float(ratio), "topk": int(edge_topk), "link": str(link_predictor_id), "bank": bank_manifest.get("nested_bank_id", "")}
    )
    write_json(manifest_path, manifest)
    return manifest
