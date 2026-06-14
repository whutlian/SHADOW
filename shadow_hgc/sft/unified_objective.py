from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store


def classwise_sqrt_budget(labels: torch.Tensor, train_rows: torch.Tensor, total_budget: int, num_classes: int) -> dict[int, int]:
    train_labels = labels[train_rows].to(torch.long)
    counts = torch.bincount(train_labels.clamp_min(0), minlength=int(num_classes)).to(torch.float64)
    active = [cls for cls in range(int(num_classes)) if int(counts[cls].item()) > 0]
    if not active:
        return {}
    budget = min(int(total_budget), int(train_rows.numel()))
    budget = max(min(len(active), int(train_rows.numel())), budget)
    weights = {cls: math.sqrt(float(counts[cls].item())) for cls in active}
    denom = sum(weights.values())
    raw = {cls: float(budget) * weights[cls] / max(denom, 1e-12) for cls in active}
    alloc = {cls: max(1, int(math.floor(raw[cls]))) for cls in active}
    while sum(alloc.values()) < budget:
        cls = max(active, key=lambda item: (raw[item] - math.floor(raw[item]), weights[item], -item))
        alloc[cls] += 1
        raw[cls] = math.floor(raw[cls])
    while sum(alloc.values()) > budget:
        cls = max((item for item in active if alloc[item] > 1), key=lambda item: (alloc[item], weights[item], -item), default=None)
        if cls is None:
            break
        alloc[cls] -= 1
    return alloc


def _bucket_rarity(values: np.ndarray, *, buckets: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    if float(values.max(initial=0.0)) == float(values.min(initial=0.0)):
        return np.ones(values.shape[0], dtype=np.float32)
    cuts = np.quantile(values, np.linspace(0.0, 1.0, int(buckets) + 1)[1:-1])
    bucket = np.searchsorted(cuts, values, side="right")
    counts = np.bincount(bucket, minlength=int(buckets)).astype(np.float64)
    rarity = 1.0 / np.sqrt(np.maximum(counts[bucket], 1.0))
    return (rarity / max(float(rarity.max(initial=1.0)), 1e-12)).astype(np.float32)


def _feature_matrix_for_train(feature_values: np.ndarray, train_rows: torch.Tensor) -> np.ndarray:
    values = np.asarray(feature_values, dtype=np.float32)
    rows = train_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.shape[0] == rows.shape[0]:
        return values
    if values.shape[0] > int(rows.max(initial=0)):
        return np.asarray(values[rows], dtype=np.float32)
    raise ValueError("feature_values must be aligned to train_rows or contain all node rows")


def build_unified_ranked_prefixes(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    feature_values: np.ndarray,
    budgets: list[int] | tuple[int, ...],
    num_classes: int,
    seed: int,
    selection_weights: dict[str, float],
    teacher_probs: np.ndarray | None = None,
) -> dict[int, torch.Tensor]:
    requested = sorted({max(1, int(value)) for value in budgets})
    if not requested:
        return {}
    rows = train_rows.detach().cpu().to(torch.long)
    features = _feature_matrix_for_train(feature_values, rows)
    max_budget = min(max(requested), int(rows.numel()))
    train_labels = labels[rows].detach().cpu().numpy().astype(np.int64, copy=False)
    alloc = classwise_sqrt_budget(labels, rows, max_budget, int(num_classes))
    if not alloc:
        return {budget: rows[: min(budget, rows.numel())].clone() for budget in requested}

    norms = np.linalg.norm(features, axis=1)
    coverage = _bucket_rarity(norms, buckets=16)
    class_counts = np.bincount(np.maximum(train_labels, 0), minlength=int(num_classes)).astype(np.float64)
    rare = 1.0 / np.sqrt(np.maximum(class_counts[np.maximum(train_labels, 0)], 1.0))
    rare = rare / max(float(rare.max(initial=1.0)), 1e-12)
    rng = np.random.default_rng(int(seed))
    diversity = rng.uniform(0.0, 1.0, size=rows.numel())
    soft = np.zeros(rows.numel(), dtype=np.float32)
    boundary = np.zeros(rows.numel(), dtype=np.float32)
    if teacher_probs is not None:
        probs = _feature_matrix_for_train(np.asarray(teacher_probs, dtype=np.float32), rows)
        probs = np.maximum(probs, 0.0)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        confidence = probs.max(axis=1)
        top2 = np.sort(probs, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0] if top2.shape[1] == 2 else confidence
        soft = confidence.astype(np.float32)
        boundary = (1.0 - np.clip(margin, 0.0, 1.0)).astype(np.float32)
    weights = {str(k): float(v) for k, v in selection_weights.items()}
    score = (
        weights.get("coverage", 0.0) * coverage
        + weights.get("hard", 0.0)
        + weights.get("soft", 0.0) * soft
        + weights.get("boundary", 0.0) * boundary
        + weights.get("rare", 0.0) * rare
        + weights.get("diversity", 0.0) * diversity
    )

    queues: dict[int, list[int]] = {}
    for cls in sorted(alloc):
        local = np.flatnonzero(train_labels == int(cls))
        order = local[np.argsort(-score[local], kind="mergesort")]
        queues[int(cls)] = [int(idx) for idx in order.tolist()]

    pointers = {cls: 0 for cls in queues}
    used_per_class = {cls: 0 for cls in queues}
    selected_local: list[int] = []
    while len(selected_local) < max_budget:
        candidates = [cls for cls in queues if pointers[cls] < len(queues[cls])]
        if not candidates:
            break
        cls = min(
            candidates,
            key=lambda item: (
                used_per_class[item] / max(1, alloc.get(item, 1)),
                -alloc.get(item, 0),
                item,
            ),
        )
        selected_local.append(queues[cls][pointers[cls]])
        pointers[cls] += 1
        used_per_class[cls] += 1

    selected_rows = rows[torch.tensor(selected_local, dtype=torch.long)] if selected_local else rows[:0]
    return {budget: selected_rows[: min(int(budget), int(selected_rows.numel()))].clone() for budget in requested}


def select_unified_prefixes_from_memmap(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    manifest_dir: str | Path,
    budgets: list[int] | tuple[int, ...],
    num_classes: int,
    seed: int,
    selection_weights: dict[str, float],
    feature_block: str = "X0",
    teacher_probs_path: str | Path | None = None,
) -> dict[int, torch.Tensor]:
    store = load_manifest_block_store(manifest_dir).subset([feature_block])
    key = "self" if str(feature_block) == "X0" else str(feature_block).lower()
    row_np = train_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    features = np.asarray(store.arrays[key][row_np], dtype=np.float32)
    teacher_probs = None
    if teacher_probs_path not in {"", None}:
        teacher_probs = np.load(Path(teacher_probs_path), mmap_mode="r")
    return build_unified_ranked_prefixes(
        labels=labels,
        train_rows=train_rows,
        feature_values=features,
        budgets=budgets,
        num_classes=int(num_classes),
        seed=int(seed),
        selection_weights=selection_weights,
        teacher_probs=teacher_probs,
    )
