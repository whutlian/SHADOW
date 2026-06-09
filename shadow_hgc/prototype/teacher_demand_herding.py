from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherDemandHerdingResult:
    indices: torch.Tensor
    labels: torch.Tensor
    diagnostics: dict


def _class_budgets(labels: torch.Tensor, total_budget: int) -> dict[int, int]:
    classes, counts = torch.unique(labels, return_counts=True)
    weights = counts.to(torch.float64).sqrt()
    weights = weights / weights.sum().clamp_min(1e-12)
    raw = weights * int(total_budget)
    budgets = {int(cls.item()): max(1, int(torch.floor(value).item())) for cls, value in zip(classes, raw)}
    while sum(budgets.values()) > int(total_budget):
        key = max(budgets, key=lambda item: budgets[item])
        if budgets[key] > 1:
            budgets[key] -= 1
        else:
            break
    while sum(budgets.values()) < int(total_budget):
        key = int(classes[int(torch.argmax(raw - torch.floor(raw)).item())].item())
        budgets[key] += 1
    return budgets


def select_teacher_demand_herding(
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    total_budget: int,
    teacher_valid: bool,
    uncertainty: torch.Tensor | None = None,
    central_fraction: float = 0.70,
    diversity_fraction: float = 0.20,
    boundary_fraction: float = 0.10,
    seed: int = 0,
) -> TeacherDemandHerdingResult:
    del seed
    train_idx = train_idx.to(torch.long)
    train_labels = labels[train_idx].to(torch.long)
    train_embeddings = embeddings[train_idx].to(torch.float32)
    budgets = _class_budgets(train_labels, int(total_budget))
    selected: list[int] = []
    central_count = 0
    diversity_count = 0
    boundary_count = 0
    for class_id, budget in budgets.items():
        local = torch.nonzero(train_labels == class_id, as_tuple=False).flatten()
        if local.numel() == 0 or budget <= 0:
            continue
        class_emb = train_embeddings[local]
        centroid = class_emb.mean(dim=0, keepdim=True)
        distances = torch.cdist(class_emb, centroid).flatten()
        central_n = min(int(round(budget * float(central_fraction))), budget)
        if not teacher_valid:
            central_n = min(budget, central_n + max(0, int(round(budget * float(boundary_fraction)))))
        central_order = torch.argsort(distances)
        chosen_local: list[int] = []
        for pos in central_order.tolist():
            if len(chosen_local) >= central_n:
                break
            chosen_local.append(pos)
        central_count += len(chosen_local)
        remaining_budget = budget - len(chosen_local)
        remaining = [idx for idx in range(local.numel()) if idx not in set(chosen_local)]
        diversity_n = min(remaining_budget, int(round(budget * float(diversity_fraction))))
        if diversity_n > 0 and remaining:
            far_order = sorted(remaining, key=lambda pos: float(distances[pos]), reverse=True)
            for pos in far_order[:diversity_n]:
                chosen_local.append(pos)
            diversity_count += min(diversity_n, len(far_order))
        remaining_budget = budget - len(chosen_local)
        if teacher_valid and remaining_budget > 0 and uncertainty is not None:
            unc = uncertainty[train_idx[local]].to(torch.float32)
            boundary_order = torch.argsort(unc, descending=True)
            chosen_set = set(chosen_local)
            for pos in boundary_order.tolist():
                if pos in chosen_set:
                    continue
                chosen_local.append(pos)
                boundary_count += 1
                if len(chosen_local) >= budget:
                    break
        if len(chosen_local) < budget:
            chosen_set = set(chosen_local)
            for pos in central_order.tolist():
                if pos not in chosen_set:
                    chosen_local.append(pos)
                if len(chosen_local) >= budget:
                    break
        selected.extend(int(train_idx[local[pos]].item()) for pos in chosen_local[:budget])
    selected_tensor = torch.tensor(selected[: int(total_budget)], dtype=torch.long)
    selected_labels = labels[selected_tensor].to(torch.long) if selected_tensor.numel() else torch.empty(0, dtype=torch.long)
    entropy = 0.0
    if selected_labels.numel():
        hist = torch.bincount(selected_labels.clamp_min(0)).to(torch.float32)
        probs = hist / hist.sum().clamp_min(1.0)
        entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
    return TeacherDemandHerdingResult(
        indices=selected_tensor,
        labels=selected_labels,
        diagnostics={
            "prototype_mode": "teacher_demand_herding",
            "herding_central_count": int(central_count),
            "herding_diversity_count": int(diversity_count),
            "herding_boundary_count": int(boundary_count if teacher_valid else 0),
            "teacher_used_for_herding": bool(teacher_valid),
            "prototype_selection_embedding_dim": int(embeddings.shape[1]),
            "prototype_selection_entropy_stats": {"selected_label_entropy": entropy},
        },
    )

