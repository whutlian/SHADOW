from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from shadow_hgc.sft.coreset import select_classwise_coreset_rows


@dataclass(frozen=True)
class STCInitResult:
    row_ids: np.ndarray
    z_init: np.ndarray
    y_init: np.ndarray
    weights: np.ndarray
    init_method: str


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _balanced_budget(train_labels: np.ndarray, m: int) -> dict[int, int]:
    classes, counts = np.unique(train_labels.astype(np.int64), return_counts=True)
    if classes.size == 0:
        return {}
    m = max(int(m), int(classes.size))
    budget = {int(cls): int(m // classes.size) for cls in classes}
    for cls in classes[: m % classes.size]:
        budget[int(cls)] += 1
    return budget


def class_balanced_random_init(
    table: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    train_idx: np.ndarray | torch.Tensor,
    *,
    m: int,
    seed: int,
    init_method: str = "class_balanced_random",
) -> STCInitResult:
    table_np = _to_numpy(table).astype(np.float32, copy=False)
    labels_np = _to_numpy(labels).astype(np.int64, copy=False)
    train_idx_np = _to_numpy(train_idx).astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    budget = _balanced_budget(labels_np[train_idx_np], int(m))
    selected: list[np.ndarray] = []
    for cls, count in budget.items():
        cls_rows = train_idx_np[labels_np[train_idx_np] == int(cls)]
        if cls_rows.size == 0:
            continue
        replace = cls_rows.size < int(count)
        selected.append(rng.choice(cls_rows, size=int(count), replace=replace))
    if not selected:
        row_ids = train_idx_np[:0]
    else:
        row_ids = np.concatenate(selected).astype(np.int64, copy=False)
    if row_ids.size > int(m):
        row_ids = row_ids[: int(m)]
    while row_ids.size < int(m) and train_idx_np.size:
        row_ids = np.concatenate([row_ids, rng.choice(train_idx_np, size=1, replace=True).astype(np.int64)])
    return STCInitResult(
        row_ids=row_ids,
        z_init=table_np[row_ids].astype(np.float32, copy=True),
        y_init=labels_np[row_ids].astype(np.int64, copy=True),
        weights=np.ones(row_ids.shape[0], dtype=np.float32),
        init_method=init_method,
    )


def random_init(
    table: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    train_idx: np.ndarray | torch.Tensor,
    *,
    m: int,
    seed: int,
) -> STCInitResult:
    table_np = _to_numpy(table).astype(np.float32, copy=False)
    labels_np = _to_numpy(labels).astype(np.int64, copy=False)
    train_idx_np = _to_numpy(train_idx).astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    replace = train_idx_np.size < int(m)
    row_ids = rng.choice(train_idx_np, size=int(m), replace=replace).astype(np.int64)
    return STCInitResult(row_ids, table_np[row_ids].copy(), labels_np[row_ids].copy(), np.ones(int(m), np.float32), "random")


def coreset_init(
    signature: np.ndarray | torch.Tensor,
    table: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    train_idx: np.ndarray | torch.Tensor,
    *,
    m: int,
    mode: str,
    seed: int,
) -> STCInitResult:
    labels_t = torch.as_tensor(_to_numpy(labels), dtype=torch.long)
    train_t = torch.as_tensor(_to_numpy(train_idx), dtype=torch.long)
    signature_t = torch.as_tensor(_to_numpy(signature), dtype=torch.float32)
    if signature_t.shape[0] == labels_t.shape[0]:
        signature_t = signature_t[train_t]
    rows = select_classwise_coreset_rows(signature_t, labels_t, train_t, int(m), mode=mode, seed=int(seed))
    rows_np = rows.cpu().numpy().astype(np.int64, copy=False)
    table_np = _to_numpy(table).astype(np.float32, copy=False)
    labels_np = _to_numpy(labels).astype(np.int64, copy=False)
    return STCInitResult(rows_np, table_np[rows_np].copy(), labels_np[rows_np].copy(), np.ones(rows_np.size, np.float32), mode)


def medoid_init(*args, **kwargs) -> STCInitResult:
    return coreset_init(*args, mode="medoid", **kwargs)


def kcenter_init(*args, **kwargs) -> STCInitResult:
    return coreset_init(*args, mode="kcenter", **kwargs)


def products_uca_hybrid_mixup_init(
    signature: np.ndarray | torch.Tensor,
    table: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    train_idx: np.ndarray | torch.Tensor,
    *,
    m: int,
    seed: int,
) -> STCInitResult:
    return coreset_init(signature, table, labels, train_idx, m=m, mode="hybrid", seed=seed)
