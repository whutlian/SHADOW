from __future__ import annotations

import numpy as np


def year_scalar_feature(years: np.ndarray, *, train_mask: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=np.float32)
    train_years = years[np.asarray(train_mask, dtype=bool)]
    mean = float(train_years.mean()) if train_years.size else float(years.mean())
    std = float(train_years.std()) if train_years.size else float(years.std())
    std = max(std, 1e-6)
    return ((years - mean) / std).reshape(-1, 1).astype(np.float32)


def year_bucket_onehot(years: np.ndarray, *, train_mask: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=np.int64)
    train_years = years[np.asarray(train_mask, dtype=bool)]
    min_year = int(train_years.min()) if train_years.size else int(years.min())
    max_year = int(train_years.max()) if train_years.size else int(years.max())
    buckets = np.clip(years, min_year, max_year) - min_year
    out = np.zeros((years.shape[0], max_year - min_year + 1), dtype=np.float32)
    out[np.arange(years.shape[0]), buckets] = 1.0
    return out


def temporal_labelreuse_decay(
    edge_index: np.ndarray,
    years: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    num_classes: int,
    gamma: float,
) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    years = np.asarray(years, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    out = np.zeros((years.shape[0], int(num_classes)), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    keep = train_mask[src]
    for s, d in zip(src[keep], dst[keep]):
        cls = int(labels[s])
        if 0 <= cls < int(num_classes):
            lag = max(0, int(years[d]) - int(years[s]))
            out[d, cls] += float(np.exp(-float(gamma) * lag))
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sum, out=out, where=row_sum > 0)
    return out


def apply_arxiv_teacher_gate(row: dict) -> dict:
    guarded = dict(row)
    acc = float(guarded.get("accuracy") or guarded.get("valid_acc") or 0.0)
    guarded["A1_passed"] = bool(acc >= 0.715)
    guarded["A2_passed"] = bool(acc >= 0.725)
    guarded["A3_passed"] = bool(acc >= 0.740)
    guarded["teacher_gate_status"] = "A1_passed" if guarded["A1_passed"] else "blocked_below_A1"
    if not guarded["A1_passed"] and str(guarded.get("method", "")).startswith("arxiv_random"):
        guarded["promotion_allowed"] = False
        guarded["promotion_status"] = "blocked_teacher_gate"
        guarded["failure_reason"] = "arxiv_teacher_below_0.715"
    return guarded
