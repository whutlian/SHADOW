from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.prototype.budgets import class_wise_budget
from shadow_hgc.prototype.cluster import PrototypeResult


@dataclass
class CoverageMedoidResult:
    indices: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    assignment: torch.Tensor
    diagnostics: dict


def _allocate_class_budget(labels: torch.Tensor, total_budget: int) -> dict[int, int]:
    classes, counts = torch.unique(labels, return_counts=True)
    total_budget = max(int(classes.numel()), int(total_budget))
    weights = torch.sqrt(counts.to(torch.float32))
    raw = total_budget * weights / weights.sum().clamp_min(1.0)
    budgets = {int(c.item()): max(1, int(round(v.item()))) for c, v in zip(classes, raw)}
    while sum(budgets.values()) > total_budget:
        key = max((k for k in budgets if budgets[k] > 1), key=lambda k: budgets[k], default=None)
        if key is None:
            break
        budgets[key] -= 1
    while sum(budgets.values()) < total_budget:
        key = int(classes[torch.argmax(counts)].item())
        budgets[key] += 1
    return budgets


def _stats(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def _select_class_medoids(
    *,
    class_rows: torch.Tensor,
    signatures: torch.Tensor,
    base_scores: torch.Tensor,
    budget: int,
    redundancy_weight: float,
) -> list[int]:
    if budget <= 0 or class_rows.numel() == 0:
        return []
    budget = min(int(budget), int(class_rows.numel()))
    selected: list[int] = []
    class_features = torch.nn.functional.normalize(signatures[class_rows].to(torch.float32), dim=1)
    for _ in range(budget):
        best_row = None
        best_score = None
        for local_pos, row in enumerate(class_rows.tolist()):
            if row in selected:
                continue
            score = float(base_scores[row].item())
            if selected:
                selected_local = torch.tensor(
                    [int((class_rows == item).nonzero(as_tuple=False).flatten()[0].item()) for item in selected],
                    dtype=torch.long,
                    device=class_rows.device,
                )
                sim = torch.matmul(class_features[local_pos], class_features[selected_local].T).max()
                score -= float(redundancy_weight) * float(sim.item())
            if best_score is None or score > best_score or (score == best_score and row < best_row):
                best_row = row
                best_score = score
        if best_row is not None:
            selected.append(int(best_row))
    return selected


def select_coverage_medoids(
    signatures: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_idx: torch.Tensor,
    class_budget: dict[int, int] | None = None,
    total_budget: int | None = None,
    coverage_scores: torch.Tensor | None = None,
    boundary_scores: torch.Tensor | None = None,
    lambda_center: float = 1.0,
    lambda_coverage: float = 1.0,
    lambda_boundary: float = 0.5,
    lambda_lad: float = 0.0,
    lambda_redundancy: float = 0.2,
    seed: int = 42,
) -> CoverageMedoidResult:
    del lambda_lad, seed
    if signatures.ndim != 2:
        raise ValueError("signatures must be rank-2")
    train_idx = train_idx.to(dtype=torch.long)
    train_labels = labels[train_idx].to(dtype=torch.long)
    if class_budget is None:
        if total_budget is None:
            raise ValueError("class_budget or total_budget is required")
        class_budget = _allocate_class_budget(train_labels, int(total_budget))

    coverage = torch.zeros(signatures.shape[0], dtype=torch.float32, device=signatures.device)
    boundary = torch.zeros(signatures.shape[0], dtype=torch.float32, device=signatures.device)
    if coverage_scores is not None:
        coverage = coverage_scores.to(device=signatures.device, dtype=torch.float32)
    if boundary_scores is not None:
        boundary = boundary_scores.to(device=signatures.device, dtype=torch.float32)

    base_scores = torch.zeros(signatures.shape[0], dtype=torch.float32, device=signatures.device)
    selected: list[int] = []
    for class_id, budget in sorted(class_budget.items()):
        class_rows = train_idx[train_labels == int(class_id)]
        if class_rows.numel() == 0:
            continue
        centroid = signatures[class_rows].to(torch.float32).mean(dim=0, keepdim=True)
        distance = torch.linalg.norm(signatures[class_rows].to(torch.float32) - centroid, dim=1)
        center_score = -distance
        base_scores[class_rows] = (
            float(lambda_center) * center_score
            + float(lambda_coverage) * coverage[class_rows]
            + float(lambda_boundary) * boundary[class_rows]
        )
        selected.extend(
            _select_class_medoids(
                class_rows=class_rows,
                signatures=signatures,
                base_scores=base_scores,
                budget=int(budget),
                redundancy_weight=float(lambda_redundancy),
            )
        )

    if not selected:
        empty = torch.empty(0, dtype=torch.long, device=signatures.device)
        return CoverageMedoidResult(empty, empty, empty.to(torch.float32), empty, {"prototype_mode": "coverage_medoid"})

    selected_idx = torch.tensor(selected, dtype=torch.long, device=signatures.device)
    medoid_features = signatures[selected_idx].to(torch.float32)
    train_features = signatures[train_idx].to(torch.float32)
    dist = torch.cdist(train_features, medoid_features)
    assignment = dist.argmin(dim=1)
    weights = torch.bincount(assignment, minlength=selected_idx.numel()).to(torch.float32)
    diagnostics = {
        "prototype_mode": "coverage_medoid",
        "selected_medoids_per_class": {
            str(class_id): int((labels[selected_idx] == int(class_id)).sum().item())
            for class_id in sorted(class_budget)
        },
        "coverage_score_stats": _stats(coverage[train_idx]),
        "boundary_score_stats": _stats(boundary[train_idx]),
        "redundancy_score_stats": {"weight": float(lambda_redundancy)},
        "medoid_real_node_ratio": 1.0,
    }
    return CoverageMedoidResult(
        indices=selected_idx.cpu(),
        labels=labels[selected_idx].to(torch.long).cpu(),
        weights=weights.cpu(),
        assignment=assignment.cpu(),
        diagnostics=diagnostics,
    )


def coverage_medoid_prototypes(
    *,
    phi_target: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    M_tau: int,
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    strict_budget: bool = False,
    coverage_scores: torch.Tensor | None = None,
    boundary_scores: torch.Tensor | None = None,
    seed: int = 42,
) -> tuple[PrototypeResult, dict]:
    budget = class_wise_budget(
        labels,
        train_idx,
        int(M_tau),
        min_proto_per_class=min_proto_per_class,
        budget_alpha=budget_alpha,
        strict=strict_budget,
    )
    result = select_coverage_medoids(
        signatures,
        labels,
        train_idx=train_idx,
        class_budget=budget.budgets,
        coverage_scores=coverage_scores,
        boundary_scores=boundary_scores,
        seed=seed,
    )
    target_to_cell = torch.full((labels.numel(),), -1, dtype=torch.long)
    selected_idx = result.indices.to(dtype=torch.long, device=labels.device)
    assignment = result.assignment.to(dtype=torch.long, device=labels.device)
    cell_members: list[torch.Tensor] = []
    prototype_features: list[torch.Tensor] = []
    prototype_labels: list[int] = []
    prototype_weights: list[float] = []
    for cell_id, medoid_idx in enumerate(selected_idx.tolist()):
        members = train_idx[assignment == cell_id].to(torch.long)
        if members.numel() == 0:
            members = torch.tensor([medoid_idx], dtype=torch.long, device=train_idx.device)
        target_to_cell[members] = cell_id
        cell_members.append(members)
        prototype_features.append(phi_target[int(medoid_idx)])
        prototype_labels.append(int(labels[int(medoid_idx)].item()))
        prototype_weights.append(float(members.numel()))
    prototypes = PrototypeResult(
        prototype_features=torch.stack(prototype_features, dim=0),
        prototype_labels=torch.tensor(prototype_labels, dtype=torch.long),
        prototype_weights=torch.tensor(prototype_weights, dtype=torch.float32),
        target_to_cell=target_to_cell.cpu(),
        cell_members=[members.cpu() for members in cell_members],
        requested_M_tau=budget.requested_M_tau,
        effective_M_tau=len(prototype_features),
        num_classes=budget.num_classes,
        min_proto_per_class=budget.min_proto_per_class,
        budget_alpha=budget.budget_alpha,
        budget_upshifted=budget.budget_upshifted,
        class_budget=dict(budget.budgets),
    )
    return prototypes, result.diagnostics
