from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.prototype.budgets import class_wise_budget


@dataclass
class PrototypeResult:
    prototype_features: torch.Tensor
    prototype_labels: torch.Tensor
    prototype_weights: torch.Tensor
    target_to_cell: torch.Tensor
    cell_members: list[torch.Tensor]
    requested_M_tau: int
    effective_M_tau: int
    num_classes: int
    min_proto_per_class: int
    budget_alpha: float
    budget_upshifted: bool
    class_budget: dict[int, int]


def _cluster_labels(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    n = x.shape[0]
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=5,
            batch_size=max(16, min(1024, n)),
        )
        return torch.as_tensor(km.fit_predict(x.detach().cpu().numpy()), dtype=torch.long)
    except Exception:
        generator = torch.Generator().manual_seed(seed)
        centers = x[torch.randperm(n, generator=generator)[:k]].clone()
        for _ in range(8):
            assign = torch.argmin(torch.cdist(x, centers), dim=1)
            for cluster_id in range(k):
                mask = assign == cluster_id
                if bool(mask.any()):
                    centers[cluster_id] = x[mask].mean(dim=0)
        return torch.argmin(torch.cdist(x, centers), dim=1)


def class_wise_prototypes(
    *,
    phi_target: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    M_tau: int,
    signature_idx: torch.Tensor | None = None,
    min_proto_per_class: int = 1,
    budget_alpha: float = 0.5,
    strict_budget: bool = False,
    seed: int = 0,
) -> PrototypeResult:
    budget_result = class_wise_budget(
        labels,
        train_idx,
        M_tau,
        min_proto_per_class=min_proto_per_class,
        budget_alpha=budget_alpha,
        strict=strict_budget,
    )
    budgets = budget_result.budgets
    target_to_cell = torch.full((labels.numel(),), -1, dtype=torch.long)
    prototype_features: list[torch.Tensor] = []
    prototype_labels: list[int] = []
    prototype_weights: list[float] = []
    cell_members: list[torch.Tensor] = []
    if signature_idx is not None:
        signature_pos = torch.full((labels.numel(),), -1, dtype=torch.long)
        signature_pos[signature_idx.to(torch.long)] = torch.arange(signature_idx.numel(), dtype=torch.long)
    else:
        signature_pos = None

    next_cell = 0
    for class_id in sorted(budgets):
        class_train = train_idx[labels[train_idx] == class_id]
        if class_train.numel() == 0:
            continue
        k = min(budgets[class_id], class_train.numel())
        signature_rows = class_train if signature_pos is None else signature_pos[class_train]
        local_assign = _cluster_labels(signatures[signature_rows], k, seed + class_id)
        for local_cell in range(k):
            members = class_train[local_assign == local_cell].to(torch.long)
            if members.numel() == 0:
                continue
            target_to_cell[members] = next_cell
            cell_members.append(members)
            prototype_features.append(phi_target[members].mean(dim=0))
            prototype_labels.append(class_id)
            prototype_weights.append(float(members.numel()))
            next_cell += 1

    return PrototypeResult(
        prototype_features=torch.stack(prototype_features, dim=0),
        prototype_labels=torch.tensor(prototype_labels, dtype=torch.long),
        prototype_weights=torch.tensor(prototype_weights, dtype=torch.float32),
        target_to_cell=target_to_cell,
        cell_members=cell_members,
        requested_M_tau=budget_result.requested_M_tau,
        effective_M_tau=budget_result.effective_M_tau,
        num_classes=budget_result.num_classes,
        min_proto_per_class=budget_result.min_proto_per_class,
        budget_alpha=budget_result.budget_alpha,
        budget_upshifted=budget_result.budget_upshifted,
        class_budget=dict(budget_result.budgets),
    )
