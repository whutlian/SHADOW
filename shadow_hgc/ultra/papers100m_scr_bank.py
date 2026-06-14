from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_scr_materialize import ratio_count
from shadow_hgc.ultra.papers100m_teacher import load_teacher_topk_cache


SCR_POLICY_CLASS_RANDOM = "scr_class_random"
SCR_POLICY_CLASS_TEMPORAL = "scr_class_temporal_random"
SCR_POLICY_CLASS_TEMPORAL_DEGREE = "scr_class_temporal_degree_random"
SCR_POLICY_FULL = "scr_full_stochastic_coverage"
SCR_POLICY_FULL_TEACHER_WEIGHT = "scr_full_stochastic_coverage_plus_teacher_weight"

SCR_POLICIES: tuple[str, ...] = (
    SCR_POLICY_CLASS_RANDOM,
    SCR_POLICY_CLASS_TEMPORAL,
    SCR_POLICY_CLASS_TEMPORAL_DEGREE,
    SCR_POLICY_FULL,
    SCR_POLICY_FULL_TEACHER_WEIGHT,
)


@dataclass(frozen=True)
class ScrRankResult:
    global_rank: np.ndarray
    priority: np.ndarray
    base_priority: np.ndarray
    teacher_tiebreak_weight: np.ndarray
    class_label: np.ndarray
    degree_bucket: np.ndarray
    feature_bucket: np.ndarray
    year_bucket: np.ndarray
    coverage_bucket: np.ndarray
    teacher_pred_class: np.ndarray
    teacher_confidence: np.ndarray
    coverage_axes: tuple[str, ...]
    class_floor_actual_min: int
    class_floor_violation_count: int
    coverage_bucket_count: int
    empty_bucket_count: int


def policy_to_bank_policy(method: str, *, teacher_weight_eta: float = 0.10) -> str:
    method = str(method)
    if method == SCR_POLICY_FULL_TEACHER_WEIGHT:
        token = f"{float(teacher_weight_eta):.2f}".replace(".", "p")
        return f"{method}_eta{token}"
    return method


def bank_policy_to_method(policy: str) -> str:
    text = str(policy)
    if text.startswith(f"{SCR_POLICY_FULL_TEACHER_WEIGHT}_eta"):
        return SCR_POLICY_FULL_TEACHER_WEIGHT
    return text


def coverage_axes_for_policy(policy: str, *, year_available: bool) -> tuple[str, ...]:
    method = bank_policy_to_method(policy)
    axes = ["class_label"]
    if method in {SCR_POLICY_CLASS_TEMPORAL, SCR_POLICY_CLASS_TEMPORAL_DEGREE, SCR_POLICY_FULL, SCR_POLICY_FULL_TEACHER_WEIGHT} and year_available:
        axes.append("year_bucket")
    if method in {SCR_POLICY_CLASS_TEMPORAL_DEGREE, SCR_POLICY_FULL, SCR_POLICY_FULL_TEACHER_WEIGHT}:
        axes.append("degree_bucket")
    if method in {SCR_POLICY_FULL, SCR_POLICY_FULL_TEACHER_WEIGHT}:
        axes.append("feature_bucket")
    return tuple(axes)


def log2_degree_bucket(degree: np.ndarray) -> np.ndarray:
    values = np.asarray(degree, dtype=np.float32)
    return np.clip(np.floor(np.log2(np.maximum(values, 0.0) + 1.0)), 0, 31).astype(np.uint16)


def feature_lsh_bucket(
    features: np.ndarray,
    *,
    seed: int,
    lsh_dim: int = 64,
    lsh_bits: int = 16,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.uint32)
    rng = np.random.default_rng(int(seed) + 1729)
    dim = min(int(lsh_dim), x.shape[1])
    cols = rng.choice(x.shape[1], size=dim, replace=False) if dim < x.shape[1] else np.arange(x.shape[1])
    reduced = x[:, cols]
    planes = rng.standard_normal((dim, int(lsh_bits))).astype(np.float32)
    signs = reduced @ planes >= 0.0
    bucket = np.zeros(x.shape[0], dtype=np.uint32)
    for bit in range(int(lsh_bits)):
        bucket |= signs[:, bit].astype(np.uint32) << np.uint32(bit)
    return bucket


def _seeded_base_priority(size: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    u = np.clip(rng.random(int(size), dtype=np.float64), 1e-12, 1.0)
    return (-np.log(u)).astype(np.float64)


def _stable_bucket_code(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        raise ValueError("at least one coverage axis is required")
    code = np.zeros(parts[0].shape[0], dtype=np.uint64)
    for idx, part in enumerate(parts):
        values = np.asarray(part, dtype=np.uint64)
        code ^= (values + np.uint64(0x9E3779B97F4A7C15 + idx * 0x100000001B3)) * np.uint64(0xBF58476D1CE4E5B9)
        code ^= code >> np.uint64(27)
    return code


def _round_robin_rank(
    *,
    candidates: np.ndarray,
    labels: np.ndarray,
    coverage_bucket: np.ndarray,
    priority: np.ndarray,
    num_classes: int,
    max_rows: int,
    class_floor_requested: int,
    floor_schedule: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int, int]:
    selected: list[int] = []
    seen: set[int] = set()
    bucket_order: dict[int, list[int]] = {}
    class_buckets: dict[int, list[int]] = {}
    for cls in range(int(num_classes)):
        cls_ids = candidates[labels[candidates] == cls]
        if cls_ids.size == 0:
            class_buckets[cls] = []
            continue
        buckets = np.unique(coverage_bucket[cls_ids])
        ordered_buckets = sorted([int(v) for v in buckets], key=lambda b: (int(np.sum(coverage_bucket[cls_ids] == b)), b))
        class_buckets[cls] = ordered_buckets
        for bucket in ordered_buckets:
            ids = cls_ids[coverage_bucket[cls_ids] == bucket]
            order = np.lexsort((ids, priority[ids]))
            bucket_order[(cls << 48) + bucket] = [int(v) for v in ids[order]]

    bucket_cursor: dict[tuple[int, int], int] = {}

    def pop_next(cls: int) -> int | None:
        buckets = class_buckets.get(int(cls), [])
        if not buckets:
            return None
        for _ in range(len(buckets)):
            cursor_key = (int(cls), -1)
            start = bucket_cursor.get(cursor_key, 0) % len(buckets)
            bucket = buckets[start]
            bucket_cursor[cursor_key] = start + 1
            values = bucket_order.get((int(cls) << 48) + int(bucket), [])
            pos_key = (int(cls), int(bucket))
            pos = bucket_cursor.get(pos_key, 0)
            while pos < len(values) and values[pos] in seen:
                pos += 1
            bucket_cursor[pos_key] = pos
            if pos < len(values):
                bucket_cursor[pos_key] = pos + 1
                return values[pos]
        return None

    class_counts = np.zeros(int(num_classes), dtype=np.int64)
    nonempty_classes = [cls for cls in range(int(num_classes)) if np.any(labels[candidates] == cls)]

    def append_floor(floor: int) -> None:
        nonlocal class_counts
        for _ in range(max(0, int(floor))):
            advanced = False
            for cls in range(int(num_classes)):
                if len(selected) >= int(max_rows):
                    break
                if class_counts[cls] > _:
                    continue
                value = pop_next(cls)
                if value is not None and value not in seen:
                    selected.append(value)
                    seen.add(value)
                    class_counts[cls] += 1
                    advanced = True
            if not advanced:
                break

    def fill_global(limit: int) -> None:
        order = np.lexsort((candidates, priority[candidates]))
        for value_np in candidates[order]:
            if len(selected) >= min(int(limit), int(max_rows)):
                break
            value = int(value_np)
            if value not in seen:
                selected.append(value)
                seen.add(value)
                class_counts[int(labels[value])] += 1

    schedule = sorted(floor_schedule or [], key=lambda item: int(item[0]))
    if schedule:
        for limit, floor in schedule:
            append_floor(int(floor))
            fill_global(int(limit))
    else:
        append_floor(int(class_floor_requested))
    floor_min = int(class_counts[nonempty_classes].min()) if nonempty_classes and selected else 0
    floor_violations = int(sum(1 for cls in nonempty_classes if class_counts[cls] < min(int(class_floor_requested), int(np.sum(labels[candidates] == cls)))))

    if len(selected) < int(max_rows):
        fill_global(int(max_rows))
    if len(selected) < int(max_rows):
        for cls in range(int(num_classes)):
            if len(selected) >= int(max_rows):
                break
            value = pop_next(cls)
            if value is not None and value not in seen:
                selected.append(value)
                seen.add(value)
                class_counts[int(labels[value])] += 1
    return np.asarray(selected, dtype=np.uint32), floor_min, floor_violations


def build_scr_rank_from_arrays(
    *,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    degree: np.ndarray,
    features: np.ndarray,
    teacher_pred_class: np.ndarray,
    teacher_confidence: np.ndarray,
    policy: str,
    seed: int,
    max_rows: int,
    num_classes: int,
    class_floor_requested: int = 32,
    floor_schedule: list[tuple[int, int]] | None = None,
    feature_lsh_dim: int = 64,
    feature_lsh_bits: int = 16,
    teacher_weight_eta: float = 0.0,
    year_bucket: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
) -> ScrRankResult:
    del valid_labels, test_labels
    labels_arr = np.asarray(labels, dtype=np.int16)
    candidate = np.asarray(candidate_mask, dtype=bool) & (labels_arr >= 0)
    candidates = np.flatnonzero(candidate).astype(np.int64)
    if candidates.size == 0:
        raise ValueError("SCR requires at least one labeled train candidate")
    max_rows = min(int(max_rows), int(candidates.size))
    degree_bucket = log2_degree_bucket(degree)
    year_available = year_bucket is not None
    if year_bucket is None:
        year_values = np.zeros(labels_arr.shape[0], dtype=np.uint16)
    else:
        year_values = np.asarray(year_bucket, dtype=np.uint16)
    method = bank_policy_to_method(policy)
    use_feature = method in {SCR_POLICY_FULL, SCR_POLICY_FULL_TEACHER_WEIGHT}
    if use_feature:
        feature_bucket = feature_lsh_bucket(features, seed=seed, lsh_dim=feature_lsh_dim, lsh_bits=feature_lsh_bits)
    else:
        feature_bucket = np.zeros(labels_arr.shape[0], dtype=np.uint32)
    axes = coverage_axes_for_policy(policy, year_available=year_available)
    parts: list[np.ndarray] = []
    for axis in axes:
        if axis == "class_label":
            parts.append(labels_arr.astype(np.uint16))
        elif axis == "year_bucket":
            parts.append(year_values)
        elif axis == "degree_bucket":
            parts.append(degree_bucket)
        elif axis == "feature_bucket":
            parts.append(feature_bucket)
    coverage_bucket = _stable_bucket_code(parts)
    base = _seeded_base_priority(labels_arr.shape[0], seed=seed)
    order = np.lexsort((candidates, coverage_bucket[candidates], base[candidates]))
    seen_counts: dict[int, int] = {}
    coverage_weight = np.ones(labels_arr.shape[0], dtype=np.float64)
    for value_np in candidates[order]:
        value = int(value_np)
        bucket = int(coverage_bucket[value])
        seen = seen_counts.get(bucket, 0)
        coverage_weight[value] = 1.0 / math.sqrt(float(seen) + 1.0)
        seen_counts[bucket] = seen + 1
    eta = float(teacher_weight_eta) if method == SCR_POLICY_FULL_TEACHER_WEIGHT else 0.0
    teacher_weight = np.clip(1.0 + eta * np.asarray(teacher_confidence, dtype=np.float64), 0.5, 2.0)
    priority = base / np.clip(coverage_weight * teacher_weight, 0.5, 2.0)
    rank, floor_min, floor_violations = _round_robin_rank(
        candidates=candidates,
        labels=labels_arr,
        coverage_bucket=coverage_bucket,
        priority=priority,
        num_classes=int(num_classes),
        max_rows=max_rows,
        class_floor_requested=int(class_floor_requested),
        floor_schedule=floor_schedule,
    )
    observed_buckets = np.unique(coverage_bucket[candidates])
    return ScrRankResult(
        global_rank=rank,
        priority=priority.astype(np.float64),
        base_priority=base.astype(np.float64),
        teacher_tiebreak_weight=teacher_weight.astype(np.float64),
        class_label=labels_arr,
        degree_bucket=degree_bucket,
        feature_bucket=feature_bucket.astype(np.uint32),
        year_bucket=year_values,
        coverage_bucket=coverage_bucket,
        teacher_pred_class=np.asarray(teacher_pred_class, dtype=np.uint16),
        teacher_confidence=np.asarray(teacher_confidence, dtype=np.float32),
        coverage_axes=axes,
        class_floor_actual_min=floor_min,
        class_floor_violation_count=floor_violations,
        coverage_bucket_count=int(observed_buckets.size),
        empty_bucket_count=0,
    )


def _write_memmap(path: Path, values: np.ndarray, dtype: np.dtype | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.dtype(dtype))
    mm = np.memmap(path, mode="w+", dtype=np.dtype(dtype), shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


def _target_labels(cache_root: Path, manifest: dict[str, Any], target_size: int) -> np.ndarray:
    target_idx = np.memmap(cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    labels = np.memmap(cache_root / "raw" / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(int(manifest["num_nodes"]),))
    return np.asarray(labels[np.asarray(target_idx, dtype=np.int64)], dtype=np.int16)


def _train_mask(cache_root: Path, manifest: dict[str, Any], target_size: int) -> np.ndarray:
    train = np.asarray(
        np.memmap(cache_root / "raw" / "train_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest["train_size"]),)),
        dtype=np.int64,
    )
    mask = np.zeros(target_size, dtype=bool)
    mask[train] = True
    return mask


def _load_year_bucket(cache_root: Path, target_size: int) -> tuple[np.ndarray | None, bool]:
    candidates = [
        cache_root / "raw" / "year_target.i16.memmap",
        cache_root / "raw" / "paper_year.i16.memmap",
        cache_root / "raw" / "year.i16.memmap",
    ]
    for path in candidates:
        if path.exists():
            year = np.asarray(np.memmap(path, mode="r", dtype=np.int16, shape=(target_size,)), dtype=np.int32)
            if year.size:
                year = np.where(year >= 0, year, 0)
                bucket = ((year - int(year.min())) // 5).astype(np.uint16)
                return bucket, True
    return None, False


def _load_feature_block(cache_root: Path, target_size: int, feature_dim: int) -> np.ndarray:
    path = cache_root / "sft" / "X0_target.fp16.memmap"
    return np.asarray(np.memmap(path, mode="r", dtype=np.float16, shape=(target_size, feature_dim)), dtype=np.float32)


def build_scr_bank(
    ctx: Papers100MCacheContext,
    *,
    policy: str,
    seed: int = 42,
    max_ratio: float = 0.0005,
    feature_lsh_dim: int = 64,
    feature_lsh_bits: int = 16,
    degree_bucket_mode: str = "log2",
    teacher_weight_eta: float = 0.0,
    candidate_universe: str = "train_targets",
    force: bool = False,
) -> dict[str, Any]:
    if str(candidate_universe) != "train_targets":
        raise ValueError("T37 SCR bank currently promotes only train_targets candidate_universe")
    bank_dir = ctx.cache_root / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest_path = bank_dir / "bank_manifest.json"
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        existing_rows = int(existing.get("selected_max_rows", 0) or 0)
        denominator = int(existing.get("full_node_ratio_denominator", ctx.manifest.get("num_nodes", 0)))
        target_size = int(existing.get("target_universe_size", ctx.manifest.get("target_universe_size", 0)))
        required_rows = ratio_count(float(max_ratio), denominator, target_size)
        if existing_rows >= required_rows and float(existing.get("max_ratio_for_bank", existing.get("max_ratio", 0.0)) or 0.0) >= float(max_ratio):
            return existing
    ctx.assert_ready(["manifest", "sft_cache", "teacher_cache"])
    started = time.perf_counter()
    bank_dir.mkdir(parents=True, exist_ok=True)
    target_size = int(ctx.manifest["target_universe_size"])
    denominator = int(ctx.manifest["num_nodes"])
    max_rows = ratio_count(float(max_ratio), denominator, target_size)
    floor_schedule = [
        (ratio_count(0.00005, denominator, max_rows), 4),
        (ratio_count(0.00010, denominator, max_rows), 8),
        (ratio_count(0.00020, denominator, max_rows), 16),
        (ratio_count(0.00050, denominator, max_rows), 32),
    ]
    labels = _target_labels(ctx.cache_root, ctx.manifest, target_size)
    train_mask = _train_mask(ctx.cache_root, ctx.manifest, target_size)
    degree = np.asarray(
        np.memmap(ctx.cache_root / "sft" / "degree_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 2)),
        dtype=np.float32,
    ).sum(axis=1)
    method = bank_policy_to_method(policy)
    if method in {SCR_POLICY_FULL, SCR_POLICY_FULL_TEACHER_WEIGHT}:
        features = _load_feature_block(ctx.cache_root, target_size, int(ctx.manifest["feature_dim"]))
    else:
        features = np.zeros((target_size, 1), dtype=np.float32)
    teacher = load_teacher_topk_cache(ctx.cache_root)
    teacher_pred = np.asarray(teacher.topk_class_ids[:, 0], dtype=np.uint16)
    teacher_conf = np.asarray(teacher.topk_probs[:, 0], dtype=np.float32)
    year_bucket, year_available = _load_year_bucket(ctx.cache_root, target_size)
    result = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=policy,
        seed=int(seed),
        max_rows=max_rows,
        num_classes=int(ctx.manifest["num_classes"]),
        class_floor_requested=32,
        floor_schedule=floor_schedule,
        feature_lsh_dim=int(feature_lsh_dim),
        feature_lsh_bits=int(feature_lsh_bits),
        teacher_weight_eta=float(teacher_weight_eta),
        year_bucket=year_bucket,
    )
    selected = result.global_rank
    target_idx = np.memmap(ctx.cache_root / "raw" / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))
    _write_memmap(bank_dir / "global_rank.u32.memmap", selected, np.uint32)
    _write_memmap(bank_dir / "global.queue.u32.memmap", selected, np.uint32)
    _write_memmap(bank_dir / "target_local_id.u32.memmap", selected, np.uint32)
    _write_memmap(bank_dir / "node_id.u32.memmap", np.asarray(target_idx[selected], dtype=np.uint32), np.uint32)
    _write_memmap(bank_dir / "true_label_if_train.i16.memmap", labels[selected], np.int16)
    _write_memmap(bank_dir / "pred_class.u16.memmap", teacher_pred[selected], np.uint16)
    _write_memmap(bank_dir / "confidence.f16.memmap", teacher_conf[selected], np.float16)
    _write_memmap(bank_dir / "priority.f32.memmap", result.priority[selected], np.float32)
    _write_memmap(bank_dir / "degree_bucket.u16.memmap", result.degree_bucket[selected], np.uint16)
    _write_memmap(bank_dir / "feature_bucket.u32.memmap", result.feature_bucket[selected], np.uint32)
    _write_memmap(bank_dir / "year_bucket.u16.memmap", result.year_bucket[selected], np.uint16)
    _write_memmap(bank_dir / "coverage_bucket.u64.memmap", result.coverage_bucket[selected], np.uint64)
    parent_ids = ctx.cache_ids()
    manifest = {
        "dataset_name": ctx.manifest["dataset_name"],
        "selection_policy": str(policy),
        "seed": int(seed),
        "max_ratio": float(max_ratio),
        "max_ratio_for_bank": float(max_ratio),
        "full_node_ratio_denominator": denominator,
        "target_universe_size": target_size,
        "selected_max_rows": int(selected.size),
        "nested_selection": True,
        "bank_build_count": 1,
        "candidate_universe": str(candidate_universe),
        "coverage_axes": ",".join(result.coverage_axes),
        "year_bucket_available": bool(year_available),
        "degree_bucket_mode": str(degree_bucket_mode),
        "feature_bucket_mode": "lsh_sign" if "feature_bucket" in result.coverage_axes else "unused",
        "feature_lsh_dim": int(feature_lsh_dim),
        "feature_lsh_bits": int(feature_lsh_bits),
        "teacher_weight_eta": float(teacher_weight_eta),
        "uses_teacher_weighting": bank_policy_to_method(policy) == SCR_POLICY_FULL_TEACHER_WEIGHT,
        "class_floor_requested": 32,
        "class_floor_actual_min": int(result.class_floor_actual_min),
        "class_floor_violation_count": int(result.class_floor_violation_count),
        "coverage_bucket_count": int(result.coverage_bucket_count),
        "empty_bucket_count": int(result.empty_bucket_count),
        "selection_bank_time": float(time.perf_counter() - started),
        "selection_bank_bytes": directory_bytes(bank_dir),
        "uses_exact_all_pair_distance": False,
        "uses_full_class_kmeans": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "parent_cache_ids": parent_ids,
        "created_at": utc_now(),
    }
    manifest["selection_bank_id"] = stable_hash(
        {"policy": policy, "seed": seed, "max_ratio": max_ratio, "parents": parent_ids, "rows": int(selected.size), "kind": "t37_scr"}
    )
    manifest["nested_bank_id"] = manifest["selection_bank_id"]
    write_json(manifest_path, manifest)
    return manifest


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p64 = np.asarray(p, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    p64 = p64 / max(float(p64.sum()), 1e-12)
    q64 = q64 / max(float(q64.sum()), 1e-12)
    mask = p64 > 0
    return float((p64[mask] * (np.log(np.maximum(p64[mask], 1e-12)) - np.log(np.maximum(q64[mask], 1e-12)))).sum())


def audit_scr_bank(
    cache_root: str | Path,
    *,
    policy: str,
    seed: int,
    ratios: list[float],
) -> list[dict[str, Any]]:
    root = Path(cache_root) / "selection_bank" / f"policy={policy}_seed{int(seed)}"
    manifest = read_json(root / "bank_manifest.json")
    selected_max = int(manifest["selected_max_rows"])
    denominator = int(manifest["full_node_ratio_denominator"])
    target_size = int(manifest["target_universe_size"])
    num_classes = int(read_json(Path(cache_root) / "manifest.json")["num_classes"])
    selected = np.memmap(root / "global_rank.u32.memmap", mode="r", dtype=np.uint32, shape=(selected_max,))
    true_label = np.asarray(np.memmap(root / "true_label_if_train.i16.memmap", mode="r", dtype=np.int16, shape=(selected_max,)), dtype=np.int64)
    pred = np.asarray(np.memmap(root / "pred_class.u16.memmap", mode="r", dtype=np.uint16, shape=(selected_max,)), dtype=np.int64)
    coverage = np.asarray(np.memmap(root / "coverage_bucket.u64.memmap", mode="r", dtype=np.uint64, shape=(selected_max,)), dtype=np.uint64)
    confidence = np.asarray(np.memmap(root / "confidence.f16.memmap", mode="r", dtype=np.float16, shape=(selected_max,)), dtype=np.float32)
    teacher = load_teacher_topk_cache(cache_root)
    teacher_prior = np.bincount(np.asarray(teacher.topk_class_ids[:, 0], dtype=np.int64), minlength=num_classes).astype(np.float64)
    rows: list[dict[str, Any]] = []
    previous_positions: set[int] | None = None
    previous_ratio = ""
    for ratio in [float(value) for value in ratios]:
        count = ratio_count(ratio, denominator, selected_max)
        positions = set(range(count))
        overlap = "" if previous_positions is None else len(previous_positions & positions) / max(1, len(previous_positions))
        violation = 0 if previous_positions is None else len(previous_positions - positions)
        hard = true_label[:count]
        hard = hard[hard >= 0]
        hard_counts = np.bincount(hard, minlength=num_classes).astype(np.float64) if hard.size else np.zeros(num_classes, dtype=np.float64)
        pred_counts = np.bincount(pred[:count], minlength=num_classes).astype(np.float64)
        class_counts = hard_counts
        floor_min = int(class_counts.min()) if class_counts.size else 0
        if ratio <= 0.00005 + 1e-12:
            requested_floor = 4
        elif ratio <= 0.00010 + 1e-12:
            requested_floor = 8
        elif ratio <= 0.00020 + 1e-12:
            requested_floor = 16
        else:
            requested_floor = int(manifest.get("class_floor_requested", 32) or 32)
        floor_violations = int(np.sum(class_counts < requested_floor)) if requested_floor > 0 else 0
        unique_coverage = np.unique(coverage[:count])
        rows.append(
            {
                "ratio": ratio,
                "ratio_percent": ratio * 100.0,
                "method": bank_policy_to_method(policy),
                "bank_policy": policy,
                "selection_bank_id": manifest["selection_bank_id"],
                "selected_count": count,
                "target_universe_size": target_size,
                "prefix_previous_ratio": previous_ratio,
                "prefix_overlap_with_previous_ratio": overlap,
                "prefix_violation_count": violation,
                "selected_class_count": int(np.sum(hard_counts > 0)),
                "selected_predicted_class_count": int(np.sum(pred_counts > 0)),
                "selected_train_anchor_count": int(hard.size),
                "selected_soft_prior_kl": _kl(pred_counts + 1e-6, teacher_prior + 1e-6),
                "selected_hard_label_prior_kl": _kl(hard_counts + 1e-6, teacher_prior + 1e-6) if hard.size else "",
                "coverage_bucket_count": int(unique_coverage.size),
                "empty_bucket_count": int(manifest.get("empty_bucket_count", 0) or 0),
                "class_floor_requested": requested_floor,
                "class_floor_actual_min": floor_min,
                "class_floor_violation_count": floor_violations,
                "confidence_mean": float(confidence[:count].mean()) if count else 0.0,
            }
        )
        previous_positions = positions
        previous_ratio = str(ratio)
    return rows
