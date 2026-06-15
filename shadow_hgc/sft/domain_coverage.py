from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch


def _as_numpy(values: np.ndarray | torch.Tensor | Iterable[int], *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        out = values.detach().cpu().numpy()
    else:
        out = np.asarray(values)
    return out.astype(dtype, copy=False) if dtype is not None else out


def _stable_combine(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    size = int(parts[0].shape[0])
    out = np.zeros(size, dtype=np.uint64)
    for idx, part in enumerate(parts):
        values = np.asarray(part, dtype=np.uint64)
        salt = np.uint64(0x9E3779B97F4A7C15 + idx * 0x100000001B3)
        out ^= (values + salt) * np.uint64(0xBF58476D1CE4E5B9)
        out ^= out >> np.uint64(29)
    return out


def build_domain_bucket_ids(
    signatures: np.ndarray,
    *,
    train_mask: np.ndarray | torch.Tensor,
    labels: torch.Tensor | np.ndarray | None = None,
    seed: int = 42,
    projection_dim: int = 32,
    num_quantiles: int = 8,
    degree: np.ndarray | None = None,
    teacher_pred_class: np.ndarray | None = None,
    teacher_confidence: np.ndarray | None = None,
) -> np.ndarray:
    del labels
    x = np.asarray(signatures, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("signatures must be a 2D array")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.uint64)
    mask = _as_numpy(train_mask, dtype=bool)
    if mask.shape[0] != x.shape[0]:
        raise ValueError("train_mask must align with signatures")
    rng = np.random.default_rng(int(seed) + 4001)
    dim = min(max(1, int(projection_dim)), x.shape[1])
    cols = rng.choice(x.shape[1], size=dim, replace=False) if dim < x.shape[1] else np.arange(x.shape[1])
    planes = rng.standard_normal((dim, min(8, dim)), dtype=np.float32)
    projected = x[:, cols] @ planes
    fit = projected[mask] if np.any(mask) else projected
    parts: list[np.ndarray] = []
    q_count = max(2, int(num_quantiles))
    for col in range(projected.shape[1]):
        cuts = np.quantile(fit[:, col], np.linspace(0.0, 1.0, q_count + 1)[1:-1])
        parts.append(np.searchsorted(cuts, projected[:, col], side="right").astype(np.uint64))
    if degree is not None:
        deg = np.asarray(degree, dtype=np.float32)
        parts.append(np.clip(np.floor(np.log2(np.maximum(deg, 0.0) + 1.0)), 0, 31).astype(np.uint64))
    if teacher_pred_class is not None:
        parts.append(np.asarray(teacher_pred_class, dtype=np.uint64))
    if teacher_confidence is not None:
        conf = np.asarray(teacher_confidence, dtype=np.float32)
        parts.append(np.clip(np.floor(conf * 10.0), 0, 10).astype(np.uint64))
    return _stable_combine(parts)


def _mass(bucket_ids: np.ndarray, indices: np.ndarray | list[int] | tuple[int, ...] | None = None) -> dict[int, float]:
    buckets = np.asarray(bucket_ids, dtype=np.uint64)
    if indices is not None:
        buckets = buckets[np.asarray(indices, dtype=np.int64)]
    if buckets.size == 0:
        return {}
    unique, counts = np.unique(buckets, return_counts=True)
    denom = float(counts.sum())
    return {int(bucket): float(count) / denom for bucket, count in zip(unique.tolist(), counts.tolist())}


def domain_coverage_gap(bucket_ids: np.ndarray, selected_indices: np.ndarray | list[int] | tuple[int, ...]) -> float:
    all_mass = _mass(bucket_ids)
    selected_mass = _mass(bucket_ids, selected_indices)
    keys = set(all_mass) | set(selected_mass)
    return 0.5 * sum(abs(float(selected_mass.get(key, 0.0)) - float(all_mass.get(key, 0.0))) for key in keys)


def domain_undercoverage_scores(bucket_ids: np.ndarray, selected_indices: np.ndarray | list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    buckets = np.asarray(bucket_ids, dtype=np.uint64)
    if buckets.size == 0:
        return np.zeros(0, dtype=np.float32)
    all_mass = _mass(buckets)
    selected_mass = _mass(buckets, selected_indices) if selected_indices is not None else {}
    scores = np.asarray([max(0.0, all_mass.get(int(bucket), 0.0) - selected_mass.get(int(bucket), 0.0)) for bucket in buckets], dtype=np.float32)
    max_value = float(scores.max(initial=0.0))
    if max_value > 0.0:
        scores = scores / max_value
    return scores.astype(np.float32)


def domain_train_all_undercoverage_scores(bucket_ids: np.ndarray, train_rows: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    buckets = np.asarray(bucket_ids, dtype=np.uint64)
    if buckets.size == 0:
        return np.zeros(0, dtype=np.float32)
    rows = np.asarray(train_rows, dtype=np.int64)
    all_mass = _mass(buckets)
    train_mass = _mass(buckets, rows)
    raw = np.asarray([max(0.0, all_mass.get(int(bucket), 0.0) - train_mass.get(int(bucket), 0.0)) for bucket in buckets], dtype=np.float32)
    max_value = float(raw.max(initial=0.0))
    if max_value > 0.0:
        raw = raw / max_value
    return raw.astype(np.float32)


def selected_prior_kl(labels: np.ndarray | torch.Tensor, selected_indices: np.ndarray | list[int] | tuple[int, ...], *, num_classes: int) -> float:
    values = _as_numpy(labels, dtype=np.int64)
    selected = values[np.asarray(selected_indices, dtype=np.int64)]
    selected = selected[selected >= 0]
    if selected.size == 0:
        return 0.0
    p = np.bincount(selected, minlength=int(num_classes)).astype(np.float64) + 1e-6
    q = np.bincount(values[values >= 0], minlength=int(num_classes)).astype(np.float64) + 1e-6
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum())


def reservoir_cache_id(dataset: str, *, seed: int, policy: str, max_budget: int, domain_gap: float) -> str:
    payload = f"{dataset}|{int(seed)}|{policy}|{int(max_budget)}|{float(domain_gap):.6f}".encode("utf-8")
    return "t40_" + hashlib.sha1(payload).hexdigest()[:16]
