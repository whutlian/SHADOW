from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CandidateEdges:
    edge_index: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OperatorMatchResult:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    diagnostics: dict[str, Any]


def apply_sparse_operator(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    values = x.to(torch.float32)
    edge_index = edge_index.to(device=values.device, dtype=torch.long)
    edge_weight = edge_weight.to(device=values.device, dtype=values.dtype)
    out = torch.zeros_like(values)
    if edge_index.numel() > 0:
        out.index_add_(0, edge_index[1], values[edge_index[0]] * edge_weight.unsqueeze(1))
    return out


def edge_weights_by_dst_softmax(edge_index: torch.Tensor, logits: torch.Tensor, *, num_nodes: int, tau: float = 1.0) -> torch.Tensor:
    edge_index = edge_index.to(torch.long)
    scores = logits.to(torch.float32) / max(float(tau), 1e-6)
    weight = torch.zeros_like(scores)
    for dst in range(int(num_nodes)):
        mask = edge_index[1] == dst
        if bool(mask.any()):
            weight[mask] = torch.softmax(scores[mask], dim=0)
    return weight


def _renormalize_by_dst(edge_index: torch.Tensor, weight: torch.Tensor, *, num_nodes: int) -> torch.Tensor:
    out = weight.to(torch.float32).clamp_min(0.0).clone()
    for dst in range(int(num_nodes)):
        mask = edge_index[1] == dst
        if bool(mask.any()):
            out[mask] = out[mask] / out[mask].sum().clamp_min(1e-12)
    return out


def project_topk_by_dst(edge_index: torch.Tensor, weight: torch.Tensor, *, num_nodes: int, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = edge_index.to(torch.long).cpu()
    weight = weight.to(torch.float32).cpu()
    kept_pairs: list[tuple[int, int]] = []
    kept_weight: list[float] = []
    k = max(1, int(topk))
    for dst in range(int(num_nodes)):
        idx = torch.nonzero(edge_index[1] == dst, as_tuple=False).view(-1)
        if idx.numel() == 0:
            kept_pairs.append((dst, dst))
            kept_weight.append(1.0)
            continue
        local = idx[torch.argsort(weight[idx], descending=True)[:k]]
        for pos in local.tolist():
            kept_pairs.append((int(edge_index[0, pos].item()), int(dst)))
            kept_weight.append(float(weight[pos].item()))
    out_index = torch.tensor(kept_pairs, dtype=torch.long).t().contiguous()
    out_weight = torch.tensor(kept_weight, dtype=torch.float32)
    return out_index, _renormalize_by_dst(out_index, out_weight, num_nodes=int(num_nodes))


def build_knn_candidate_edges(features: torch.Tensor, *, candidate_topk: int, include_self: bool = True) -> CandidateEdges:
    x = torch.nn.functional.normalize(features.detach().to(torch.float32).cpu(), p=2, dim=1)
    n = int(x.shape[0])
    if n == 0:
        return CandidateEdges(torch.empty((2, 0), dtype=torch.long), {"uses_dense_adjacency": False, "uses_exact_pairwise": False})
    k = min(max(1, int(candidate_topk)), max(1, n - 1))
    score = x @ x.t()
    score.fill_diagonal_(-float("inf"))
    _, idx = torch.topk(score, k=k, dim=1)
    pairs: list[tuple[int, int]] = []
    for dst in range(n):
        if include_self:
            pairs.append((dst, dst))
        for src in idx[dst].tolist():
            if int(src) != dst:
                pairs.append((int(src), dst))
    edge_index = torch.tensor(sorted(set(pairs)), dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    return CandidateEdges(
        edge_index=edge_index,
        metadata={
            "candidate_topk_per_row": int(candidate_topk),
            "operator_candidate_edges": int(edge_index.shape[1]),
            "uses_dense_adjacency": False,
            "uses_exact_pairwise": False,
            "uses_full_edge_index_on_gpu": False,
        },
    )


def _operator_entropy(edge_index: torch.Tensor, weight: torch.Tensor, num_nodes: int) -> float:
    ent = 0.0
    rows = 0
    for dst in range(int(num_nodes)):
        local = weight[edge_index[1] == dst]
        if local.numel():
            ent += float((-(local.clamp_min(1e-12) * local.clamp_min(1e-12).log()).sum()).item())
            rows += 1
    return ent / max(1, rows)


def _row_sum_error(edge_index: torch.Tensor, weight: torch.Tensor, num_nodes: int) -> float:
    worst = 0.0
    for dst in range(int(num_nodes)):
        local = weight[edge_index[1] == dst]
        if local.numel():
            worst = max(worst, abs(float(local.sum().item()) - 1.0))
    return worst


def fit_operator_match(
    *,
    x0: torch.Tensor,
    x1_target: torch.Tensor,
    candidate_edge_index: torch.Tensor,
    topk: int,
    steps: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    tau: float = 1.0,
    lambda_x1: float = 1.0,
    lambda_x2: float = 0.5,
    x2_target: torch.Tensor | None = None,
    y0: torch.Tensor | None = None,
    y1_target: torch.Tensor | None = None,
    y2_target: torch.Tensor | None = None,
    lambda_y1: float = 0.25,
    lambda_y2: float = 0.10,
) -> OperatorMatchResult:
    torch.manual_seed(int(seed))
    x0 = x0.to(torch.float32)
    x1_target = x1_target.to(torch.float32)
    edge_index = candidate_edge_index.to(torch.long)
    num_nodes = int(x0.shape[0])
    logits = torch.zeros(edge_index.shape[1], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([logits], lr=float(lr))
    final_losses = {"operator_loss_x1": 0.0, "operator_loss_x2": 0.0, "operator_loss_y1": 0.0, "operator_loss_y2": 0.0}
    for _ in range(max(0, int(steps))):
        optimizer.zero_grad()
        weight = edge_weights_by_dst_softmax(edge_index, logits, num_nodes=num_nodes, tau=float(tau))
        px0 = apply_sparse_operator(x0, edge_index, weight)
        loss_x1 = torch.mean((px0 - x1_target) ** 2)
        loss = float(lambda_x1) * loss_x1
        loss_x2 = torch.tensor(0.0)
        if x2_target is not None and float(lambda_x2) != 0.0:
            p2x0 = apply_sparse_operator(px0, edge_index, weight)
            loss_x2 = torch.mean((p2x0 - x2_target.to(torch.float32)) ** 2)
            loss = loss + float(lambda_x2) * loss_x2
        loss_y1 = torch.tensor(0.0)
        loss_y2 = torch.tensor(0.0)
        if y0 is not None and y1_target is not None and float(lambda_y1) != 0.0:
            py0 = apply_sparse_operator(y0.to(torch.float32), edge_index, weight)
            loss_y1 = torch.mean((py0 - y1_target.to(torch.float32)) ** 2)
            loss = loss + float(lambda_y1) * loss_y1
            if y2_target is not None and float(lambda_y2) != 0.0:
                p2y0 = apply_sparse_operator(py0, edge_index, weight)
                loss_y2 = torch.mean((p2y0 - y2_target.to(torch.float32)) ** 2)
                loss = loss + float(lambda_y2) * loss_y2
        loss.backward()
        optimizer.step()
        final_losses = {
            "operator_loss_x1": float(loss_x1.detach().item()),
            "operator_loss_x2": float(loss_x2.detach().item()),
            "operator_loss_y1": float(loss_y1.detach().item()),
            "operator_loss_y2": float(loss_y2.detach().item()),
        }
    with torch.no_grad():
        weight = edge_weights_by_dst_softmax(edge_index, logits.detach(), num_nodes=num_nodes, tau=float(tau))
        projected_index, projected_weight = project_topk_by_dst(edge_index, weight, num_nodes=num_nodes, topk=int(topk))
    row_error = _row_sum_error(projected_index, projected_weight, num_nodes)
    negative = int((projected_weight < -1e-12).sum().item())
    return OperatorMatchResult(
        edge_index=projected_index,
        edge_weight=projected_weight,
        diagnostics={
            **final_losses,
            "operator_topk": int(topk),
            "operator_candidate_edges": int(edge_index.shape[1]),
            "operator_edges": int(projected_index.shape[1]),
            "operator_row_sum_error": float(row_error),
            "operator_negative_weight_count": negative,
            "operator_entropy": float(_operator_entropy(projected_index, projected_weight, num_nodes)),
            "uses_dense_adjacency": False,
            "uses_exact_pairwise": False,
        },
    )


def synthetic_operator_targets(num_nodes: int, feature_dim: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    x0 = torch.randn(int(num_nodes), int(feature_dim), generator=generator)
    rolled = torch.roll(x0, shifts=1, dims=0)
    return x0, rolled
