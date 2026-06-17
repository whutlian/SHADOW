from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.sft.domain_coverage import domain_train_all_undercoverage_scores, domain_undercoverage_scores
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
    domain_bucket_ids: np.ndarray | None = None,
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
    domain = np.zeros(rows.numel(), dtype=np.float32)
    if domain_bucket_ids is not None:
        all_domain = np.asarray(domain_bucket_ids)
        row_np = rows.numpy()
        if all_domain.shape[0] == rows.numel():
            domain_buckets = all_domain
            domain = domain_undercoverage_scores(domain_buckets).astype(np.float32)
        elif all_domain.shape[0] > int(row_np.max(initial=0)):
            domain_buckets = all_domain[row_np]
            domain = domain_train_all_undercoverage_scores(all_domain, row_np)[row_np].astype(np.float32)
        else:
            raise ValueError("domain_bucket_ids must be aligned to train_rows or contain all node rows")
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
        + weights.get("domain", 0.0) * domain
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


def _teacher_components(teacher_probs: np.ndarray | None, rows: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    soft = np.zeros(rows.numel(), dtype=np.float32)
    boundary = np.zeros(rows.numel(), dtype=np.float32)
    if teacher_probs is None:
        return soft, boundary
    probs = _feature_matrix_for_train(np.asarray(teacher_probs, dtype=np.float32), rows)
    probs = np.maximum(probs, 0.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    confidence = probs.max(axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0] if top2.shape[1] == 2 else confidence
    return confidence.astype(np.float32), (1.0 - np.clip(margin, 0.0, 1.0)).astype(np.float32)


def _domain_component(domain_bucket_ids: np.ndarray | None, rows: torch.Tensor) -> np.ndarray:
    domain = np.zeros(rows.numel(), dtype=np.float32)
    if domain_bucket_ids is None:
        return domain
    all_domain = np.asarray(domain_bucket_ids)
    row_np = rows.numpy()
    if all_domain.shape[0] == rows.numel():
        return domain_undercoverage_scores(all_domain).astype(np.float32)
    if all_domain.shape[0] > int(row_np.max(initial=0)):
        return domain_train_all_undercoverage_scores(all_domain, row_np)[row_np].astype(np.float32)
    raise ValueError("domain_bucket_ids must be aligned to train_rows or contain all node rows")


def _score_components(
    *,
    labels: torch.Tensor,
    rows: torch.Tensor,
    features: np.ndarray,
    num_classes: int,
    seed: int,
    teacher_probs: np.ndarray | None,
    domain_bucket_ids: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_labels = labels[rows].detach().cpu().numpy().astype(np.int64, copy=False)
    norms = np.linalg.norm(features, axis=1)
    coverage = _bucket_rarity(norms, buckets=16)
    class_counts = np.bincount(np.maximum(train_labels, 0), minlength=int(num_classes)).astype(np.float64)
    rare = 1.0 / np.sqrt(np.maximum(class_counts[np.maximum(train_labels, 0)], 1.0))
    rare = rare / max(float(rare.max(initial=1.0)), 1e-12)
    rng = np.random.default_rng(int(seed))
    diversity = rng.uniform(0.0, 1.0, size=rows.numel()).astype(np.float32)
    soft, boundary = _teacher_components(teacher_probs, rows)
    domain = _domain_component(domain_bucket_ids, rows)
    return train_labels, {
        "coverage": coverage.astype(np.float32),
        "domain": domain.astype(np.float32),
        "soft": soft.astype(np.float32),
        "boundary": boundary.astype(np.float32),
        "rare": rare.astype(np.float32),
        "diversity": diversity.astype(np.float32),
        "hard": np.ones(rows.numel(), dtype=np.float32),
    }


def _weighted_component_score(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(next(iter(components.values())), dtype=np.float32)
    for key, values in components.items():
        score = score + float(weights.get(key, 0.0)) * values
    return score


def _weights_for_budget(stage_selection_weights: dict[int, dict[str, float]], budget: int) -> dict[str, float]:
    if int(budget) in stage_selection_weights:
        return {str(k): float(v) for k, v in stage_selection_weights[int(budget)].items()}
    eligible = [int(value) for value in stage_selection_weights if int(value) <= int(budget)]
    if eligible:
        return {str(k): float(v) for k, v in stage_selection_weights[max(eligible)].items()}
    smallest = min(int(value) for value in stage_selection_weights)
    return {str(k): float(v) for k, v in stage_selection_weights[smallest].items()}


def build_staged_ranked_prefixes(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    feature_values: np.ndarray,
    budgets: list[int] | tuple[int, ...],
    num_classes: int,
    seed: int,
    stage_selection_weights: dict[int, dict[str, float]],
    teacher_probs: np.ndarray | None = None,
    domain_bucket_ids: np.ndarray | None = None,
) -> dict[int, torch.Tensor]:
    requested = sorted({max(1, int(value)) for value in budgets})
    if not requested:
        return {}
    if not stage_selection_weights:
        raise ValueError("stage_selection_weights cannot be empty")
    rows = train_rows.detach().cpu().to(torch.long)
    features = _feature_matrix_for_train(feature_values, rows)
    max_budget = min(max(requested), int(rows.numel()))
    train_labels, components = _score_components(
        labels=labels,
        rows=rows,
        features=features,
        num_classes=int(num_classes),
        seed=int(seed),
        teacher_probs=teacher_probs,
        domain_bucket_ids=domain_bucket_ids,
    )

    selected_local: list[int] = []
    selected_set: set[int] = set()
    selected_per_class = {cls: 0 for cls in range(int(num_classes))}
    prefixes: dict[int, torch.Tensor] = {}
    for budget in requested:
        target_budget = min(int(budget), max_budget)
        alloc = classwise_sqrt_budget(labels, rows, target_budget, int(num_classes))
        if not alloc:
            prefixes[int(budget)] = rows[:target_budget].clone()
            continue
        weights = _weights_for_budget(stage_selection_weights, int(budget))
        score = _weighted_component_score(components, weights)
        queues: dict[int, list[int]] = {}
        for cls in sorted(alloc):
            local = np.flatnonzero(train_labels == int(cls))
            order = local[np.argsort(-score[local], kind="mergesort")]
            queues[int(cls)] = [int(idx) for idx in order.tolist() if int(idx) not in selected_set]
        pointers = {cls: 0 for cls in queues}
        while len(selected_local) < target_budget:
            candidates = [
                cls
                for cls in queues
                if pointers[cls] < len(queues[cls]) and selected_per_class.get(cls, 0) < int(alloc.get(cls, 0))
            ]
            if not candidates:
                candidates = [cls for cls in queues if pointers[cls] < len(queues[cls])]
            if not candidates:
                break
            cls = min(
                candidates,
                key=lambda item: (
                    selected_per_class.get(item, 0) / max(1, int(alloc.get(item, 1))),
                    -int(alloc.get(item, 0)),
                    item,
                ),
            )
            local_idx = int(queues[cls][pointers[cls]])
            pointers[cls] += 1
            if local_idx in selected_set:
                continue
            selected_local.append(local_idx)
            selected_set.add(local_idx)
            selected_per_class[cls] = selected_per_class.get(cls, 0) + 1
        selected_tensor = rows[torch.tensor(selected_local[:target_budget], dtype=torch.long)] if selected_local else rows[:0]
        prefixes[int(budget)] = selected_tensor.clone()
    return prefixes


def select_unified_prefixes_from_memmap(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    manifest_dir: str | Path,
    budgets: list[int] | tuple[int, ...],
    num_classes: int,
    seed: int,
    selection_weights: dict[str, float],
    stage_selection_weights: dict[int, dict[str, float]] | None = None,
    feature_block: str = "X0",
    teacher_probs_path: str | Path | None = None,
    domain_bucket_ids: np.ndarray | None = None,
) -> dict[int, torch.Tensor]:
    store = load_manifest_block_store(manifest_dir).subset([feature_block])
    key = "self" if str(feature_block) == "X0" else str(feature_block).lower()
    row_np = train_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    features = np.asarray(store.arrays[key][row_np], dtype=np.float32)
    teacher_probs = None
    if teacher_probs_path not in {"", None}:
        teacher_probs = np.load(Path(teacher_probs_path), mmap_mode="r")
    if stage_selection_weights:
        return build_staged_ranked_prefixes(
            labels=labels,
            train_rows=train_rows,
            feature_values=features,
            budgets=budgets,
            num_classes=int(num_classes),
            seed=int(seed),
            stage_selection_weights=stage_selection_weights,
            teacher_probs=teacher_probs,
            domain_bucket_ids=domain_bucket_ids,
        )
    return build_unified_ranked_prefixes(
        labels=labels,
        train_rows=train_rows,
        feature_values=features,
        budgets=budgets,
        num_classes=int(num_classes),
        seed=int(seed),
        selection_weights=selection_weights,
        teacher_probs=teacher_probs,
        domain_bucket_ids=domain_bucket_ids,
    )


def select_staged_prefixes_from_memmap(
    *,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    manifest_dir: str | Path,
    budgets: list[int] | tuple[int, ...],
    num_classes: int,
    seed: int,
    stage_selection_weights: dict[int, dict[str, float]],
    feature_block: str = "X0",
    teacher_probs_path: str | Path | None = None,
    domain_bucket_ids: np.ndarray | None = None,
) -> dict[int, torch.Tensor]:
    representative_budget = max(int(value) for value in budgets)
    return select_unified_prefixes_from_memmap(
        labels=labels,
        train_rows=train_rows,
        manifest_dir=manifest_dir,
        budgets=budgets,
        num_classes=int(num_classes),
        seed=int(seed),
        selection_weights=_weights_for_budget(stage_selection_weights, representative_budget),
        stage_selection_weights=stage_selection_weights,
        feature_block=feature_block,
        teacher_probs_path=teacher_probs_path,
        domain_bucket_ids=domain_bucket_ids,
    )
