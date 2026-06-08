from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shadow_hgc.prototype.budgets import class_wise_budget
from shadow_hgc.prototype.cluster import PrototypeResult, _cluster_labels


@dataclass
class BoundaryBudgetSplit:
    base_budget: dict[int, int]
    boundary_budget: dict[int, int]


@dataclass
class BoundaryPrototypeResult(PrototypeResult):
    base_budget: dict[int, int] | None = None
    boundary_budget: dict[int, int] | None = None
    boundary_pool_size_by_class: dict[int, int] | None = None
    num_boundary_prototypes: int = 0
    num_base_prototypes: int = 0
    boundary_score_stats: dict | None = None


def score_boundary_nodes(
    *,
    logits: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
    labels: torch.Tensor,
    method: str = "entropy",
) -> torch.Tensor:
    if method not in {"entropy", "margin", "loss"}:
        raise ValueError("method must be entropy, margin, or loss")
    if probabilities is None:
        if logits is None:
            raise ValueError("logits or probabilities are required")
        probabilities = torch.softmax(logits, dim=1)
    probs = probabilities.to(torch.float32)
    labels = labels.to(torch.long)
    if method == "entropy":
        return -(probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    if method == "loss":
        return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels.clamp_min(0), reduction="none")
    true_prob = probs.gather(1, labels.clamp_min(0).unsqueeze(1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, labels.clamp_min(0).unsqueeze(1), -1.0)
    other = masked.max(dim=1).values
    return other - true_prob


def split_boundary_budget(
    class_budget: dict[int, int],
    *,
    boundary_fraction: float,
) -> BoundaryBudgetSplit:
    base_budget = {}
    boundary_budget = {}
    for class_id, total in sorted(class_budget.items()):
        total = int(total)
        if total <= 1:
            boundary = 0
        else:
            boundary = int(round(float(boundary_fraction) * total))
            boundary = max(1, min(total - 1, boundary))
        base_budget[int(class_id)] = total - boundary
        boundary_budget[int(class_id)] = boundary
    return BoundaryBudgetSplit(base_budget=base_budget, boundary_budget=boundary_budget)


def _kcenter_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    n = x.shape[0]
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    selected = [0]
    dist = torch.cdist(x, x[[0]]).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(dist).item())
        selected.append(idx)
        dist = torch.minimum(dist, torch.cdist(x, x[[idx]]).squeeze(1))
    return torch.tensor(selected, dtype=torch.long)


def _centers_from_subset(signatures: torch.Tensor, rows: torch.Tensor, k: int, method: str, seed: int) -> torch.Tensor:
    if k <= 0 or rows.numel() == 0:
        return torch.empty(0, signatures.shape[1], dtype=signatures.dtype, device=signatures.device)
    rows = rows.to(torch.long)
    k = min(k, int(rows.numel()))
    x = signatures[rows]
    if method == "kcenter":
        return x[_kcenter_indices(x, k)]
    if method == "kmeans":
        assign = _cluster_labels(x, k, seed)
        return torch.stack([x[assign == cluster].mean(dim=0) for cluster in range(k)], dim=0)
    raise ValueError("clustering_method must be kmeans or kcenter")


def _score_stats(scores: torch.Tensor, method: str) -> dict:
    if scores.numel() == 0:
        return {"method": method, "count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "method": method,
        "count": int(scores.numel()),
        "mean": float(scores.mean().item()),
        "max": float(scores.max().item()),
        "min": float(scores.min().item()),
    }


def _assign_with_nonempty_centers(class_signatures: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(class_signatures, centers)
    assign = torch.argmin(distances, dim=1)
    if centers.shape[0] >= class_signatures.shape[0]:
        return torch.arange(class_signatures.shape[0], dtype=torch.long, device=class_signatures.device)
    available = torch.ones(class_signatures.shape[0], dtype=torch.bool, device=class_signatures.device)
    anchors: list[tuple[int, int]] = []
    for center_id in range(centers.shape[0]):
        candidate_dist = distances[:, center_id].clone()
        candidate_dist[~available] = float("inf")
        row_id = int(torch.argmin(candidate_dist).item())
        available[row_id] = False
        anchors.append((row_id, center_id))
    for row_id, center_id in anchors:
        assign[row_id] = center_id
    return assign


def boundary_aware_prototypes(
    *,
    phi_target: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    M_tau: int,
    logits: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
    boundary_scores: torch.Tensor | None = None,
    boundary_fraction: float = 0.3,
    boundary_pool_fraction: float | None = None,
    boundary_pool_quantile: float | None = None,
    boundary_score: str = "entropy",
    clustering_method: str = "kmeans",
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    seed: int = 0,
) -> BoundaryPrototypeResult:
    budget = class_wise_budget(
        labels,
        train_idx,
        M_tau,
        min_proto_per_class=min_proto_per_class,
        budget_alpha=budget_alpha,
        strict=False,
    )
    split = split_boundary_budget(budget.budgets, boundary_fraction=boundary_fraction)
    if boundary_scores is None:
        boundary_scores = score_boundary_nodes(
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            method=boundary_score,
        )
        score_method = boundary_score
    else:
        score_method = "precomputed"
    boundary_scores = boundary_scores.to(torch.float32)

    target_to_cell = torch.full((labels.numel(),), -1, dtype=torch.long)
    prototype_features: list[torch.Tensor] = []
    prototype_labels: list[int] = []
    prototype_weights: list[float] = []
    cell_members: list[torch.Tensor] = []
    boundary_pool_size_by_class: dict[int, int] = {}
    num_base = 0
    num_boundary = 0

    for class_id in sorted(budget.budgets):
        class_rows = train_idx[labels[train_idx] == class_id].to(torch.long)
        if class_rows.numel() == 0:
            continue
        base_k = min(split.base_budget[class_id], int(class_rows.numel()))
        boundary_k = min(split.boundary_budget[class_id], max(0, int(class_rows.numel()) - base_k))
        class_scores = boundary_scores[class_rows]
        if boundary_pool_quantile is not None:
            pool_fraction = max(0.0, min(1.0, 1.0 - float(boundary_pool_quantile)))
        else:
            pool_fraction = 0.4 if boundary_pool_fraction is None else float(boundary_pool_fraction)
        pool_size = max(boundary_k, int(round(pool_fraction * int(class_rows.numel())))) if boundary_k > 0 else 0
        pool_size = min(int(class_rows.numel()), pool_size)
        hard_rows = class_rows[torch.topk(class_scores, k=pool_size).indices] if pool_size > 0 else torch.empty(0, dtype=torch.long)
        boundary_pool_size_by_class[int(class_id)] = int(hard_rows.numel())

        base_centers = _centers_from_subset(signatures, class_rows, base_k, clustering_method, seed + int(class_id))
        boundary_centers = _centers_from_subset(signatures, hard_rows, boundary_k, clustering_method, seed + 997 + int(class_id))
        centers = torch.cat([base_centers, boundary_centers], dim=0)
        if centers.numel() == 0:
            continue
        local_assign = _assign_with_nonempty_centers(signatures[class_rows], centers)
        for local_cell in range(centers.shape[0]):
            members = class_rows[local_assign == local_cell].to(torch.long)
            if members.numel() == 0:
                continue
            cell_id = len(cell_members)
            target_to_cell[members] = cell_id
            cell_members.append(members)
            prototype_features.append(phi_target[members].mean(dim=0))
            prototype_labels.append(int(class_id))
            prototype_weights.append(float(members.numel()))
            if local_cell < base_centers.shape[0]:
                num_base += 1
            else:
                num_boundary += 1

    return BoundaryPrototypeResult(
        prototype_features=torch.stack(prototype_features, dim=0),
        prototype_labels=torch.tensor(prototype_labels, dtype=torch.long),
        prototype_weights=torch.tensor(prototype_weights, dtype=torch.float32),
        target_to_cell=target_to_cell,
        cell_members=cell_members,
        requested_M_tau=budget.requested_M_tau,
        effective_M_tau=len(cell_members),
        num_classes=budget.num_classes,
        min_proto_per_class=budget.min_proto_per_class,
        budget_alpha=budget.budget_alpha,
        budget_upshifted=budget.budget_upshifted,
        class_budget=dict(budget.budgets),
        base_budget=split.base_budget,
        boundary_budget=split.boundary_budget,
        boundary_pool_size_by_class=boundary_pool_size_by_class,
        num_boundary_prototypes=num_boundary,
        num_base_prototypes=num_base,
        boundary_score_stats=_score_stats(boundary_scores[train_idx], score_method),
    )
