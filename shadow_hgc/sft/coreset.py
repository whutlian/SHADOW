from __future__ import annotations

import torch


def class_sqrt_budget(labels: torch.Tensor, rows: torch.Tensor, total: int) -> dict[int, int]:
    rows = rows.to(torch.long).cpu()
    labels = labels.to(torch.long).cpu()
    y = labels[rows]
    classes = torch.unique(y, sorted=True)
    if classes.numel() == 0:
        return {}
    total = max(int(classes.numel()), int(total))
    counts = {int(cls.item()): int((y == cls).sum().item()) for cls in classes}
    weights = {cls: counts[cls] ** 0.5 for cls in counts}
    denom = sum(weights.values()) or 1.0
    budget = {cls: max(1, int(round(total * weights[cls] / denom))) for cls in counts}
    while sum(budget.values()) > total and any(value > 1 for value in budget.values()):
        candidate = max((cls for cls in budget if budget[cls] > 1), key=lambda cls: budget[cls])
        budget[candidate] -= 1
    while sum(budget.values()) < total:
        candidate = max(budget, key=lambda cls: counts[cls] / max(1, budget[cls]))
        budget[candidate] += 1
    return budget


def select_classwise_coreset_rows(
    signature: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    total: int,
    *,
    mode: str = "medoid",
    seed: int = 42,
) -> torch.Tensor:
    train_rows = train_rows.to(torch.long).cpu()
    labels = labels.to(torch.long).cpu()
    signature = signature.to(torch.float32).cpu()
    if signature.shape[0] != train_rows.numel():
        raise ValueError("signature rows must align with train_rows")
    budget = class_sqrt_budget(labels, train_rows, total)
    generator = torch.Generator().manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for cls, requested in budget.items():
        cls_pos = torch.nonzero(labels[train_rows] == int(cls), as_tuple=False).view(-1)
        if cls_pos.numel() == 0:
            continue
        k = min(int(requested), int(cls_pos.numel()))
        sig = signature[cls_pos]
        if mode in {"random", "centroid"}:
            chosen = cls_pos[torch.randperm(cls_pos.numel(), generator=generator)[:k]]
        else:
            center = sig.mean(dim=0, keepdim=True)
            dist = ((sig - center) ** 2).sum(dim=1)
            if mode == "kcenter":
                chosen_local = [int(torch.argmin(dist).item())]
                min_dist = ((sig - sig[chosen_local[0]].view(1, -1)) ** 2).sum(dim=1)
                for _ in range(1, k):
                    idx = int(torch.argmax(min_dist).item())
                    chosen_local.append(idx)
                    min_dist = torch.minimum(min_dist, ((sig - sig[idx].view(1, -1)) ** 2).sum(dim=1))
                chosen = cls_pos[torch.tensor(chosen_local, dtype=torch.long)]
            elif mode == "hybrid":
                hard = max(1, int(round(k * 0.1)))
                med = cls_pos[torch.argsort(dist)[: max(1, k - hard)]]
                far = cls_pos[torch.argsort(dist, descending=True)[:hard]]
                chosen = torch.unique(torch.cat([med, far]), sorted=False)[:k]
                if chosen.numel() < k:
                    fill = cls_pos[torch.argsort(dist)[:k]]
                    chosen = torch.unique(torch.cat([chosen, fill]), sorted=False)[:k]
            elif mode == "medoid":
                chosen = cls_pos[torch.argsort(dist)[:k]]
            else:
                raise ValueError(f"unsupported coreset mode: {mode}")
        selected.append(train_rows[chosen])
    if not selected:
        return train_rows[:0]
    return torch.cat(selected).to(torch.long)
