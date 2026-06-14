from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_teacher import load_teacher_topk_cache


NESTED_BANK_POLICY = "papers100m_stt_nested_bank_v2"


def _ratio_count(ratio: float, denominator: int, limit: int) -> int:
    return min(int(limit), max(1, int(round(float(ratio) * int(denominator)))))


def _degree_bucket(degree: np.ndarray) -> np.ndarray:
    deg = np.asarray(degree, dtype=np.float32)
    return np.clip(np.floor(np.log2(np.maximum(deg, 0.0) + 1.0)), 0, 15).astype(np.uint16)


def _safe_norm(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    mask = p > 0
    return float((p[mask] * (np.log(np.maximum(p[mask], 1e-12)) - np.log(np.maximum(q[mask], 1e-12)))).sum())


def _train_local_membership(cache_root: Path, manifest: dict[str, Any], target_size: int) -> np.ndarray:
    train = np.asarray(
        np.memmap(cache_root / "raw" / "train_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest["train_size"]),)),
        dtype=np.int64,
    )
    mask = np.zeros(target_size, dtype=bool)
    mask[train] = True
    return mask


def _true_labels_for_target(cache_root: Path, manifest: dict[str, Any], target_size: int) -> np.ndarray:
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(manifest["num_nodes"]),))
    return np.asarray(labels[np.asarray(target_idx, dtype=np.int64)], dtype=np.int16)


def _class_floor_order(
    *,
    score: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    num_classes: int,
) -> list[int]:
    anchors: list[int] = []
    for cls in range(int(num_classes)):
        candidates = np.flatnonzero(train_mask & (labels == cls))
        if candidates.size:
            best = int(candidates[np.argmax(score[candidates])])
            anchors.append(best)
    anchors.sort(key=lambda idx: float(score[idx]), reverse=True)
    return anchors


def _write_selected_memmap(path: Path, values: np.ndarray, dtype: np.dtype | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, mode="w+", dtype=np.dtype(dtype), shape=values.shape)
    mm[:] = np.asarray(values, dtype=np.dtype(dtype))
    mm.flush()
    del mm


def build_nested_bank_v2(
    ctx: Papers100MCacheContext,
    *,
    policy: str = NESTED_BANK_POLICY,
    seed: int = 7,
    max_ratio: float = 0.002,
    force: bool = False,
    teacher_id: str = "current_teacher",
) -> dict[str, Any]:
    bank_dir = ctx.cache_root / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest_path = bank_dir / "bank_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    ctx.assert_ready(["manifest", "sft_cache", "teacher_cache"])
    started = time.perf_counter()
    bank_dir.mkdir(parents=True, exist_ok=True)
    target_size = int(ctx.manifest["target_universe_size"])
    denominator = int(ctx.manifest["num_nodes"])
    max_rows = _ratio_count(float(max_ratio), denominator, target_size)
    num_classes = int(ctx.manifest["num_classes"])
    rng = np.random.default_rng(int(seed))

    teacher = load_teacher_topk_cache(ctx.cache_root)
    pred_class = np.asarray(teacher.topk_class_ids[:, 0], dtype=np.uint16)
    confidence = np.asarray(teacher.topk_probs[:, 0], dtype=np.float32)
    entropy = np.asarray(teacher.entropy[:], dtype=np.float32)
    margin = np.asarray(teacher.margin[:], dtype=np.float32)
    degree = np.asarray(
        np.memmap(ctx.cache_root / "sft" / "degree_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 2)),
        dtype=np.float32,
    ).sum(axis=1)
    bucket_id = _degree_bucket(degree)
    train_mask = _train_local_membership(ctx.cache_root, ctx.manifest, target_size)
    true_label_full = _true_labels_for_target(ctx.cache_root, ctx.manifest, target_size)
    true_label_if_train = np.full(target_size, -1, dtype=np.int16)
    true_label_if_train[train_mask] = true_label_full[train_mask]

    pred_counts = np.bincount(pred_class.astype(np.int64), minlength=num_classes).astype(np.float32)
    rare_bonus = 1.0 / np.sqrt(np.maximum(pred_counts[pred_class.astype(np.int64)], 1.0))
    rare_bonus = _safe_norm(rare_bonus)
    bucket_counts = np.bincount(bucket_id.astype(np.int64), minlength=int(bucket_id.max()) + 1).astype(np.float32)
    coverage_gain = 1.0 / np.sqrt(np.maximum(bucket_counts[bucket_id.astype(np.int64)], 1.0))
    coverage_gain = _safe_norm(coverage_gain)
    boundary_value = _safe_norm(entropy) * 0.65 + (1.0 - _safe_norm(margin)) * 0.35
    hard_anchor_quality = np.zeros(target_size, dtype=np.float32)
    correct_train = train_mask & (pred_class.astype(np.int16) == true_label_if_train)
    hard_anchor_quality[train_mask] = 0.35 + 0.35 * confidence[train_mask]
    hard_anchor_quality[correct_train] = 1.0 + confidence[correct_train]
    redundancy_penalty = _safe_norm(degree)
    score = (
        1.00 * hard_anchor_quality
        + 0.50 * confidence
        + 0.50 * confidence
        + 0.75 * coverage_gain
        + 0.50 * rare_bonus
        + 0.20 * boundary_value
        - 0.25 * redundancy_penalty
        + rng.random(target_size, dtype=np.float32) * 1e-7
    ).astype(np.float32)

    anchors = _class_floor_order(score=score, labels=true_label_if_train, train_mask=train_mask, num_classes=num_classes)
    selected: list[int] = []
    seen: set[int] = set()
    for local_id in anchors:
        if local_id not in seen and len(selected) < max_rows:
            selected.append(local_id)
            seen.add(local_id)
    order = np.argsort(-score)
    for local_id_np in order:
        local_id = int(local_id_np)
        if local_id not in seen:
            selected.append(local_id)
            seen.add(local_id)
            if len(selected) >= max_rows:
                break
    selected_arr = np.asarray(selected, dtype=np.uint32)
    target_idx = np.memmap(ctx.cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))

    _write_selected_memmap(bank_dir / "global_rank.u32.memmap", selected_arr, np.uint32)
    _write_selected_memmap(bank_dir / "global.queue.u32.memmap", selected_arr, np.uint32)
    _write_selected_memmap(bank_dir / "target_local_id.u32.memmap", selected_arr, np.uint32)
    _write_selected_memmap(bank_dir / "node_id.u32.memmap", np.asarray(target_idx[selected_arr], dtype=np.uint32), np.uint32)
    _write_selected_memmap(bank_dir / "pred_class.u16.memmap", pred_class[selected_arr], np.uint16)
    _write_selected_memmap(bank_dir / "true_label_if_train.i16.memmap", true_label_if_train[selected_arr], np.int16)
    _write_selected_memmap(bank_dir / "score.f32.memmap", score[selected_arr], np.float32)
    _write_selected_memmap(bank_dir / "hard_anchor_quality.f16.memmap", hard_anchor_quality[selected_arr], np.float16)
    _write_selected_memmap(bank_dir / "confidence.f16.memmap", confidence[selected_arr], np.float16)
    _write_selected_memmap(bank_dir / "entropy.f16.memmap", entropy[selected_arr], np.float16)
    _write_selected_memmap(bank_dir / "margin.f16.memmap", margin[selected_arr], np.float16)
    _write_selected_memmap(bank_dir / "bucket_id.u16.memmap", bucket_id[selected_arr], np.uint16)
    _write_selected_memmap(bank_dir / "class_id.u16.memmap", pred_class[selected_arr], np.uint16)
    _write_selected_memmap(bank_dir / "scores.fp32.memmap", score, np.float32)

    parent_ids = ctx.cache_ids()
    manifest = {
        "dataset_name": ctx.manifest["dataset_name"],
        "selection_policy": str(policy),
        "seed": int(seed),
        "max_ratio": float(max_ratio),
        "max_ratio_for_bank": float(max_ratio),
        "full_node_ratio_denominator": denominator,
        "target_universe_size": target_size,
        "selected_max_rows": int(selected_arr.size),
        "nested_selection": True,
        "bank_build_count": 1,
        "teacher_id": str(teacher_id),
        "teacher_cache_id": parent_ids["teacher_cache_id"],
        "bucket_core_count": int(np.sum(hard_anchor_quality[selected_arr] > 0.0)),
        "bucket_boundary_count": int(np.sum(boundary_value[selected_arr] >= np.quantile(boundary_value, 0.80))),
        "bucket_rare_count": int(np.sum(rare_bonus[selected_arr] >= np.quantile(rare_bonus, 0.80))),
        "bucket_prior_repair_count": int(np.sum(confidence[selected_arr] <= np.quantile(confidence, 0.20))),
        "bucket_hard_anchor_count": int(np.sum(train_mask[selected_arr])),
        "selection_bank_time": float(time.perf_counter() - started),
        "selection_bank_bytes": directory_bytes(bank_dir),
        "parent_cache_ids": parent_ids,
        "created_at": utc_now(),
    }
    manifest["selection_bank_id"] = stable_hash(
        {"policy": policy, "seed": seed, "max_ratio": max_ratio, "parents": parent_ids, "rows": int(selected_arr.size), "kind": "t36_nested"}
    )
    manifest["nested_bank_id"] = manifest["selection_bank_id"]
    write_json(manifest_path, manifest)
    return manifest


def load_nested_prefix(cache_root: str | Path, *, policy: str = NESTED_BANK_POLICY, seed: int = 7, ratio: float, denominator: int | None = None) -> np.ndarray:
    root = Path(cache_root) / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest = read_json(root / "bank_manifest.json")
    denom = int(denominator or manifest["full_node_ratio_denominator"])
    count = _ratio_count(float(ratio), denom, int(manifest["selected_max_rows"]))
    return np.asarray(np.memmap(root / "global_rank.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest["selected_max_rows"]),))[:count], dtype=np.uint32)


def audit_nested_bank(
    cache_root: str | Path,
    *,
    policy: str = NESTED_BANK_POLICY,
    seed: int = 7,
    ratios: list[float],
) -> list[dict[str, Any]]:
    cache_root = Path(cache_root)
    root = cache_root / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest = read_json(root / "bank_manifest.json")
    target_size = int(manifest["target_universe_size"])
    selected_max = int(manifest["selected_max_rows"])
    pred = np.memmap(root / "pred_class.u16.memmap", mode="r", dtype=np.uint16, shape=(selected_max,))
    true_label = np.memmap(root / "true_label_if_train.i16.memmap", mode="r", dtype=np.int16, shape=(selected_max,))
    confidence = np.asarray(np.memmap(root / "confidence.f16.memmap", mode="r", dtype=np.float16, shape=(selected_max,)), dtype=np.float32)
    entropy = np.asarray(np.memmap(root / "entropy.f16.memmap", mode="r", dtype=np.float16, shape=(selected_max,)), dtype=np.float32)
    bucket = np.memmap(root / "bucket_id.u16.memmap", mode="r", dtype=np.uint16, shape=(selected_max,))
    num_classes = int(read_json(cache_root / "manifest.json")["num_classes"])
    denominator = int(manifest["full_node_ratio_denominator"])
    teacher = load_teacher_topk_cache(cache_root)
    teacher_prior = np.bincount(np.asarray(teacher.topk_class_ids[:, 0], dtype=np.int64), minlength=num_classes).astype(np.float64)
    rows: list[dict[str, Any]] = []
    previous: set[int] | None = None
    previous_ratio = ""
    for ratio in [float(v) for v in ratios]:
        count = _ratio_count(ratio, denominator, selected_max)
        prefix_ids = set(range(count))
        violation_count = 0 if previous is None else len(previous - prefix_ids)
        overlap = "" if previous is None else (len(previous & prefix_ids) / max(1, len(previous)))
        pred_counts = np.bincount(np.asarray(pred[:count], dtype=np.int64), minlength=num_classes).astype(np.float64)
        hard = np.asarray(true_label[:count], dtype=np.int64)
        hard = hard[hard >= 0]
        hard_counts = np.bincount(hard, minlength=num_classes).astype(np.float64) if hard.size else np.zeros(num_classes, dtype=np.float64)
        bucket_counts = np.bincount(np.asarray(bucket[:count], dtype=np.int64), minlength=16)
        rows.append(
            {
                "ratio": ratio,
                "ratio_percent": ratio * 100.0,
                "nested_bank_id": manifest["nested_bank_id"],
                "selection_bank_id": manifest["selection_bank_id"],
                "max_ratio_for_bank": manifest["max_ratio_for_bank"],
                "selected_count": count,
                "target_universe_size": target_size,
                "prefix_previous_ratio": previous_ratio,
                "prefix_overlap_with_previous_ratio": overlap,
                "prefix_violation_count": violation_count,
                "selected_hard_label_prior_kl": _kl(hard_counts + 1e-6, teacher_prior + 1e-6) if hard.size else "",
                "selected_soft_prior_kl": _kl(pred_counts + 1e-6, teacher_prior + 1e-6),
                "selected_class_coverage": int(np.sum(pred_counts > 0)),
                "selected_predicted_class_count": int(np.sum(pred_counts > 0)),
                "selected_train_anchor_count": int(hard.size),
                "selected_boundary_count": int(np.sum(entropy[:count] >= np.quantile(entropy[:count], 0.80))) if count > 1 else 0,
                "selected_rare_count": int(np.sum(pred_counts[np.asarray(pred[:count], dtype=np.int64)] <= np.quantile(pred_counts[pred_counts > 0], 0.20))) if np.any(pred_counts > 0) else 0,
                "selected_bucket_histogram_json": ",".join(f"{idx}:{int(value)}" for idx, value in enumerate(bucket_counts) if value),
            }
        )
        previous = prefix_ids
        previous_ratio = str(ratio)
    return rows


def build_external_onecache_bank(
    ctx: Papers100MCacheContext,
    *,
    method: str,
    seed: int = 7,
    max_ratio: float = 0.00050,
    force: bool = False,
) -> dict[str, Any]:
    policy = f"{method}_t36"
    bank_dir = ctx.cache_root / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest_path = bank_dir / "bank_manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    started = time.perf_counter()
    bank_dir.mkdir(parents=True, exist_ok=True)
    target_size = int(ctx.manifest["target_universe_size"])
    denominator = int(ctx.manifest["num_nodes"])
    max_rows = _ratio_count(float(max_ratio), denominator, target_size)
    teacher = load_teacher_topk_cache(ctx.cache_root)
    confidence = np.asarray(teacher.topk_probs[:, 0], dtype=np.float32)
    degree = np.asarray(
        np.memmap(ctx.cache_root / "sft" / "degree_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 2)),
        dtype=np.float32,
    ).sum(axis=1)
    rng = np.random.default_rng(int(seed))
    if method == "random_onecache":
        order = rng.permutation(target_size)
    elif method == "herding_onecache":
        order = np.argsort(-(confidence + rng.random(target_size, dtype=np.float32) * 1e-7))
    elif method == "kcenter_onecache":
        # Scalable proxy: interleave degree buckets rather than exact all-pair k-center.
        buckets = _degree_bucket(degree)
        pieces = []
        for bucket_value in np.unique(buckets):
            ids = np.flatnonzero(buckets == bucket_value)
            rng.shuffle(ids)
            pieces.append(ids)
        order = np.concatenate(pieces) if pieces else np.arange(target_size)
    else:
        raise ValueError(f"unknown baseline method: {method}")
    selected = np.asarray(order[:max_rows], dtype=np.uint32)
    target_idx = np.memmap(ctx.cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    _write_selected_memmap(bank_dir / "global_rank.u32.memmap", selected, np.uint32)
    _write_selected_memmap(bank_dir / "global.queue.u32.memmap", selected, np.uint32)
    _write_selected_memmap(bank_dir / "target_local_id.u32.memmap", selected, np.uint32)
    _write_selected_memmap(bank_dir / "node_id.u32.memmap", np.asarray(target_idx[selected], dtype=np.uint32), np.uint32)
    parent_ids = ctx.cache_ids()
    manifest = {
        "dataset_name": ctx.manifest["dataset_name"],
        "selection_policy": policy,
        "seed": int(seed),
        "max_ratio": float(max_ratio),
        "max_ratio_for_bank": float(max_ratio),
        "full_node_ratio_denominator": denominator,
        "target_universe_size": target_size,
        "selected_max_rows": int(selected.size),
        "nested_selection": True,
        "bank_build_count": 1,
        "selection_bank_time": float(time.perf_counter() - started),
        "selection_bank_bytes": directory_bytes(bank_dir),
        "uses_exact_all_pair_distance": False,
        "notes": "kcenter_onecache is a bounded degree-bucket proxy; no exact all-pair distance",
        "parent_cache_ids": parent_ids,
        "created_at": utc_now(),
    }
    manifest["selection_bank_id"] = stable_hash({"policy": policy, "seed": seed, "max_ratio": max_ratio, "parents": parent_ids, "rows": int(selected.size)})
    manifest["nested_bank_id"] = manifest["selection_bank_id"]
    write_json(manifest_path, manifest)
    return manifest
