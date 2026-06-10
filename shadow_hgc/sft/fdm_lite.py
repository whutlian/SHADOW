from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


T25_METHODS: tuple[str, ...] = (
    "sft_hnr_random",
    "sft_hnr_fdm_herding",
    "sft_hnr_fdm_kcenter",
    "sft_hnr_fdm_hybrid",
    "sft_hnr_fdm_shadow_b1",
    "sft_hnr_fdm_shadow_b2",
)


@dataclass(frozen=True)
class CandidatePool:
    class_id: int
    subclass_id: int
    budget: int
    candidate_rows: torch.Tensor
    candidate_pos: torch.Tensor
    center: torch.Tensor


@dataclass(frozen=True)
class FDMLitePlan:
    pools: list[CandidatePool]
    class_budgets: dict[int, int]
    signature_dim: int
    num_subclasses: int
    candidate_pool_size: int
    uses_exact_pairwise: bool = False
    full_class_kmeans: bool = False


@dataclass(frozen=True)
class FDMLiteSelection:
    selected_rows: torch.Tensor
    selected_pos: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ShadowAssignment:
    src_shadow: torch.Tensor
    dst_proto: torch.Tensor
    edge_weight: torch.Tensor


def _labels_for_train_rows(labels: torch.Tensor, train_rows: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().to(torch.long).cpu()
    train_rows = train_rows.detach().to(torch.long).cpu()
    if train_rows.numel() and int(train_rows.max().item()) < int(labels.numel()):
        return labels[train_rows]
    if labels.numel() != train_rows.numel():
        raise ValueError("labels must be full-node labels or train-row-aligned labels")
    return labels


def reduce_sft_signature(signature: torch.Tensor, *, output_dim: int = 128, seed: int = 42) -> torch.Tensor:
    x = signature.detach().to(torch.float32).cpu()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    output_dim = int(output_dim)
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if x.shape[1] == output_dim:
        return x
    generator = torch.Generator().manual_seed(int(seed))
    proj = torch.randn(x.shape[1], output_dim, generator=generator, dtype=torch.float32) / math.sqrt(float(output_dim))
    return x @ proj


def allocate_t25_class_budgets(
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    *,
    total_budget: int,
    min_per_class: int = 4,
) -> dict[int, int]:
    y = _labels_for_train_rows(labels, train_rows)
    classes = torch.unique(y, sorted=True)
    if classes.numel() == 0:
        return {}
    total_budget = max(1, int(total_budget))
    min_per_class = max(0, int(min_per_class))
    counts = {int(cls.item()): int((y == cls).sum().item()) for cls in classes}
    feasible_floor = min_per_class if total_budget >= int(classes.numel()) * min_per_class else max(1, total_budget // int(classes.numel()))
    budget = {cls: min(counts[cls], feasible_floor) for cls in counts}
    remaining = max(0, total_budget - sum(budget.values()))
    while remaining > 0:
        eligible = [cls for cls in counts if budget[cls] < counts[cls]]
        if not eligible:
            break
        cls = max(eligible, key=lambda key: (math.sqrt(max(1, counts[key])) / max(1, budget[key]), counts[key]))
        budget[cls] += 1
        remaining -= 1
    while sum(budget.values()) > total_budget and any(value > 1 for value in budget.values()):
        cls = max((key for key in budget if budget[key] > 1), key=lambda key: budget[key])
        budget[cls] -= 1
    return budget


def _subclass_count(n: int, budget: int, *, fdm_k_min: int, fdm_k_max: int, beta_k: float) -> int:
    if n <= 0 or budget <= 0:
        return 0
    k = int(math.ceil(float(beta_k) * math.sqrt(float(n))))
    k = max(int(fdm_k_min), min(int(fdm_k_max), k))
    return max(1, min(k, int(n), max(1, int(budget))))


def _allocate_subclass_budgets(counts: list[int], total: int) -> list[int]:
    nonzero = [idx for idx, value in enumerate(counts) if value > 0]
    if not nonzero or total <= 0:
        return [0 for _ in counts]
    budgets = [0 for _ in counts]
    if total >= len(nonzero):
        for idx in nonzero:
            budgets[idx] = 1
    remaining = max(0, int(total) - sum(budgets))
    weights = [math.sqrt(max(1, value)) for value in counts]
    while remaining > 0:
        idx = max(nonzero, key=lambda item: weights[item] / max(1, budgets[item]))
        budgets[idx] += 1
        remaining -= 1
    return budgets


def _weighted_pool_positions(pos: torch.Tensor, weights: torch.Tensor, *, pool_size: int, seed: int) -> torch.Tensor:
    if pos.numel() <= int(pool_size):
        return pos
    generator = torch.Generator().manual_seed(int(seed))
    u = torch.rand(pos.numel(), generator=generator).clamp_min(1e-12)
    priority = -torch.log(u) / weights[pos].to(torch.float32).clamp_min(1e-6)
    keep = torch.argsort(priority)[: int(pool_size)]
    return pos[keep]


def build_fdm_lite_plan(
    signature: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    *,
    total_budget: int,
    node_weight: torch.Tensor | None = None,
    scale_bucket: str = "medium",
    min_per_class: int = 4,
    fdm_k_min: int = 2,
    fdm_k_max: int = 32,
    beta_k: float = 0.25,
    candidate_rho: int = 16,
    candidate_max: int = 1024,
    seed: int = 42,
) -> FDMLitePlan:
    x = signature.detach().to(torch.float32).cpu()
    train_rows = train_rows.detach().to(torch.long).cpu()
    y = _labels_for_train_rows(labels, train_rows)
    if x.shape[0] != train_rows.numel() or y.numel() != train_rows.numel():
        raise ValueError("signature, labels, and train_rows must align")
    if str(scale_bucket) == "ultra":
        fdm_k_max = min(int(fdm_k_max), 32)
        candidate_max = min(int(candidate_max), 1024)
    weights = torch.ones(x.shape[0], dtype=torch.float32) if node_weight is None else node_weight.detach().to(torch.float32).cpu()
    budgets = allocate_t25_class_budgets(y, torch.arange(y.numel()), total_budget=int(total_budget), min_per_class=int(min_per_class))
    pools: list[CandidatePool] = []
    max_pool_seen = 0
    for cls, class_budget in budgets.items():
        cls_pos = torch.nonzero(y == int(cls), as_tuple=False).view(-1)
        if cls_pos.numel() == 0 or class_budget <= 0:
            continue
        k_sub = _subclass_count(int(cls_pos.numel()), int(class_budget), fdm_k_min=fdm_k_min, fdm_k_max=fdm_k_max, beta_k=beta_k)
        cls_sig = x[cls_pos]
        order = torch.argsort(cls_sig[:, 0])
        sorted_pos = cls_pos[order]
        chunks = torch.chunk(sorted_pos, k_sub)
        counts = [int(chunk.numel()) for chunk in chunks]
        sub_budgets = _allocate_subclass_budgets(counts, int(class_budget))
        for sub_id, sub_pos in enumerate(chunks):
            if sub_pos.numel() == 0 or sub_budgets[sub_id] <= 0:
                continue
            budget = int(min(sub_budgets[sub_id], int(sub_pos.numel())))
            pool_size = min(int(candidate_max), int(candidate_rho) * max(1, budget), int(sub_pos.numel()))
            candidate_pos = _weighted_pool_positions(sub_pos, weights, pool_size=pool_size, seed=int(seed) + int(cls) * 1009 + sub_id)
            center = x[sub_pos].mean(dim=0)
            max_pool_seen = max(max_pool_seen, int(candidate_pos.numel()))
            pools.append(
                CandidatePool(
                    class_id=int(cls),
                    subclass_id=int(sub_id),
                    budget=budget,
                    candidate_rows=train_rows[candidate_pos],
                    candidate_pos=candidate_pos,
                    center=center,
                )
            )
    return FDMLitePlan(
        pools=pools,
        class_budgets=budgets,
        signature_dim=int(x.shape[1]),
        num_subclasses=len(pools),
        candidate_pool_size=max_pool_seen,
    )


def _append_unique(out: list[int], values: list[int], limit: int) -> None:
    seen = set(out)
    for value in values:
        if len(out) >= limit:
            return
        if value not in seen:
            out.append(int(value))
            seen.add(int(value))


def _select_from_pool(x: torch.Tensor, pool: CandidatePool, method: str, strata: list[str] | None, seed: int) -> torch.Tensor:
    pos = pool.candidate_pos
    budget = min(int(pool.budget), int(pos.numel()))
    if budget <= 0:
        return pos[:0]
    sig = x[pos]
    dist = ((sig - pool.center.view(1, -1)) ** 2).sum(dim=1)
    if method == "sft_hnr_random":
        generator = torch.Generator().manual_seed(int(seed) + pool.class_id * 917 + pool.subclass_id)
        return pos[torch.randperm(pos.numel(), generator=generator)[:budget]]
    if method == "sft_hnr_fdm_herding":
        return pos[torch.argsort(dist)[:budget]]
    if method == "sft_hnr_fdm_kcenter":
        local = [int(torch.argmin(dist).item())]
        min_dist = ((sig - sig[local[0]].view(1, -1)) ** 2).sum(dim=1)
        for _ in range(1, budget):
            idx = int(torch.argmax(min_dist).item())
            local.append(idx)
            min_dist = torch.minimum(min_dist, ((sig - sig[idx].view(1, -1)) ** 2).sum(dim=1))
        return pos[torch.tensor(local, dtype=torch.long)]
    if method in {"sft_hnr_fdm_hybrid", "sft_hnr_fdm_shadow_b1", "sft_hnr_fdm_shadow_b2"}:
        selected: list[int] = []
        near_k = max(1, int(round(0.50 * budget)))
        _append_unique(selected, pos[torch.argsort(dist)[:near_k]].tolist(), budget)
        far_k = max(1, int(round(0.25 * budget)))
        _append_unique(selected, pos[torch.argsort(dist, descending=True)[:far_k]].tolist(), budget)
        if strata is not None:
            h0 = [int(p.item()) for p in pos if strata[int(p.item())] == "H0"]
            _append_unique(selected, h0, budget)
        _append_unique(selected, pos[torch.argsort(dist)].tolist(), budget)
        return torch.tensor(selected, dtype=torch.long)
    raise ValueError(f"unsupported T25 FDM-lite method: {method}")


def select_fdm_lite_rows(
    signature: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    *,
    total_budget: int,
    method: str,
    node_weight: torch.Tensor | None = None,
    stratum: list[str] | None = None,
    fdm_signature_dim: int | None = None,
    seed: int = 42,
    scale_bucket: str = "medium",
    candidate_rho: int = 16,
    candidate_max: int = 1024,
    fdm_k_min: int = 2,
    fdm_k_max: int = 32,
) -> FDMLiteSelection:
    if method not in T25_METHODS:
        raise ValueError(f"unsupported T25 method: {method}")
    x = signature.detach().to(torch.float32).cpu()
    if fdm_signature_dim is not None and int(fdm_signature_dim) < x.shape[1]:
        x = reduce_sft_signature(x, output_dim=int(fdm_signature_dim), seed=int(seed))
    plan = build_fdm_lite_plan(
        x,
        labels,
        train_rows,
        total_budget=int(total_budget),
        node_weight=node_weight,
        scale_bucket=scale_bucket,
        fdm_k_min=fdm_k_min,
        fdm_k_max=fdm_k_max,
        candidate_rho=candidate_rho,
        candidate_max=candidate_max,
        seed=seed,
    )
    selected_pos: list[torch.Tensor] = []
    method_for_selection = "sft_hnr_fdm_hybrid" if method in {"sft_hnr_fdm_shadow_b1", "sft_hnr_fdm_shadow_b2"} else method
    for pool in plan.pools:
        selected_pos.append(_select_from_pool(x, pool, method_for_selection, stratum, seed))
    if selected_pos:
        pos = torch.unique(torch.cat(selected_pos).to(torch.long), sorted=False)
    else:
        pos = torch.empty(0, dtype=torch.long)
    if pos.numel() > int(total_budget):
        pos = pos[: int(total_budget)]
    train_rows = train_rows.detach().to(torch.long).cpu()
    diagnostics = {
        "fdm_mode": "lite",
        "fdm_signature_dim": int(x.shape[1]),
        "fdm_num_subclasses": int(plan.num_subclasses),
        "fdm_candidate_pool_size": int(plan.candidate_pool_size),
        "uses_exact_pairwise": False,
        "full_class_kmeans": False,
    }
    return FDMLiteSelection(selected_rows=train_rows[pos], selected_pos=pos, diagnostics=diagnostics)


def assign_shadow_b2(residual: torch.Tensor, shadows: torch.Tensor, *, eps: float = 1e-12) -> ShadowAssignment:
    residual = residual.detach().to(torch.float32).cpu()
    shadows = shadows.detach().to(torch.float32).cpu()
    if residual.ndim != 2 or shadows.ndim != 2:
        raise ValueError("residual and shadows must be matrices")
    if residual.shape[1] != shadows.shape[1]:
        raise ValueError("residual and shadows must share feature dimension")
    if shadows.shape[0] == 0:
        return ShadowAssignment(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32))
    k = min(2, int(shadows.shape[0]))
    dist = torch.cdist(residual, shadows, p=2)
    values, indices = torch.topk(dist, k=k, largest=False, dim=1)
    inv = 1.0 / values.clamp_min(float(eps))
    weights = inv / inv.sum(dim=1, keepdim=True).clamp_min(float(eps))
    dst = torch.arange(residual.shape[0], dtype=torch.long).repeat_interleave(k)
    return ShadowAssignment(
        src_shadow=indices.reshape(-1).to(torch.long),
        dst_proto=dst,
        edge_weight=weights.reshape(-1).clamp_min(0.0).to(torch.float32),
    )
