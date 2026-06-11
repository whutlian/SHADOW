from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuotientOperatorResult:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    diagnostics: dict[str, Any]


def _row_entropy(rows: list[list[tuple[int, float]]]) -> float:
    total = 0.0
    count = 0
    for entries in rows:
        if not entries:
            continue
        count += 1
        for _, weight in entries:
            w = max(float(weight), 1e-12)
            total -= w * math.log(w)
    return total / max(1, count)


def _finalize_rows(raw_rows: list[dict[int, float]], *, topk: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    before = sum(len(row) for row in raw_rows)
    zero_rows = 0
    repaired_rows = 0
    kept: list[list[tuple[int, float]]] = []
    for dst, row in enumerate(raw_rows):
        if not row:
            zero_rows += 1
            repaired_rows += 1
            row = {dst: 1.0}
        ordered = sorted(row.items(), key=lambda item: (-float(item[1]), int(item[0])))[: max(1, int(topk))]
        denom = sum(max(0.0, float(weight)) for _, weight in ordered)
        if denom <= 0.0:
            repaired_rows += 1
            ordered = [(dst, 1.0)]
            denom = 1.0
        kept.append([(int(src), max(0.0, float(weight)) / denom) for src, weight in ordered])
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for dst, entries in enumerate(kept):
        for src, weight in entries:
            pairs.append((src, dst))
            weights.append(float(weight))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    row_sums = torch.zeros(len(raw_rows), dtype=torch.float32)
    if edge_index.numel():
        row_sums.index_add_(0, edge_index[1], edge_weight)
    nonzero = edge_weight[edge_weight > 0]
    diagnostics = {
        "operator_edges_before_topk": int(before + zero_rows),
        "operator_edges_after_topk": int(edge_index.shape[1]),
        "operator_row_sum_error": float((row_sums - 1.0).abs().max().item()) if row_sums.numel() else 0.0,
        "operator_zero_rows": int(zero_rows),
        "operator_repaired_rows": int(repaired_rows),
        "operator_entropy": float(_row_entropy(kept)),
        "operator_max_weight": float(nonzero.max().item()) if nonzero.numel() else 0.0,
        "operator_min_nonzero_weight": float(nonzero.min().item()) if nonzero.numel() else 0.0,
        "uses_dense_adjacency": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "uses_exact_pairwise": False,
    }
    return edge_index, edge_weight, diagnostics


def build_quotient_operator(
    *,
    edge_index: torch.Tensor,
    assignments: torch.Tensor,
    num_codewords: int,
    topk: int,
    edge_weight: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    mode: str = "original_dest_normalized",
    eps: float = 1e-12,
) -> QuotientOperatorResult:
    started = time.perf_counter()
    if mode not in {"original_dest_normalized", "code_row_normalized_fallback"}:
        raise ValueError(f"unsupported quotient build mode: {mode}")
    edges = edge_index.to(torch.long).cpu()
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E] with source row first")
    assign = assignments.to(torch.long).cpu()
    num_nodes = int(assign.numel())
    raw_w = torch.ones(edges.shape[1], dtype=torch.float32) if edge_weight is None else edge_weight.to(torch.float32).cpu()
    nweight = torch.ones(num_nodes, dtype=torch.float32) if node_weight is None else node_weight.to(torch.float32).cpu()
    code_dest_mass = torch.zeros(int(num_codewords), dtype=torch.float32)
    valid_assign = (assign >= 0) & (assign < int(num_codewords))
    code_dest_mass.index_add_(0, assign[valid_assign], nweight[valid_assign])
    rows: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(int(num_codewords))]
    full_edge_scans = 1
    deg_in = torch.ones(num_nodes, dtype=torch.float32)
    if mode == "original_dest_normalized":
        full_edge_scans = 2
        deg_in = torch.zeros(num_nodes, dtype=torch.float32)
        dst = edges[1]
        valid_dst = (dst >= 0) & (dst < num_nodes)
        deg_in.index_add_(0, dst[valid_dst], raw_w[valid_dst].clamp_min(0.0))
    for pos in range(edges.shape[1]):
        src = int(edges[0, pos].item())
        dst = int(edges[1, pos].item())
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            continue
        cu = int(assign[src].item())
        cv = int(assign[dst].item())
        if cu < 0 or cv < 0 or cu >= int(num_codewords) or cv >= int(num_codewords):
            continue
        if mode == "original_dest_normalized":
            alpha = float(raw_w[pos].item()) / max(float(deg_in[dst].item()), float(eps))
            contribution = float(nweight[dst].item()) * alpha / max(float(code_dest_mass[cv].item()), float(eps))
        else:
            contribution = float(raw_w[pos].item())
        rows[cv][cu] += contribution
    if mode == "code_row_normalized_fallback":
        for row in rows:
            denom = sum(max(0.0, value) for value in row.values())
            if denom > 0.0:
                for src in list(row):
                    row[src] = max(0.0, row[src]) / denom
    edge_index_out, edge_weight_out, diagnostics = _finalize_rows(rows, topk=int(topk))
    diagnostics.update(
        {
            "operator_topk": int(topk),
            "quotient_build_mode": mode,
            "operator_mode": "sparse_codeword_quotient",
            "operator_build_time": float(time.perf_counter() - started),
            "full_edge_scans": int(full_edge_scans),
        }
    )
    return QuotientOperatorResult(edge_index=edge_index_out, edge_weight=edge_weight_out, diagnostics=diagnostics)


def quotient_to_dense(edge_index: torch.Tensor, edge_weight: torch.Tensor, *, num_codewords: int) -> torch.Tensor:
    dense = torch.zeros(int(num_codewords), int(num_codewords), dtype=torch.float32)
    if edge_index.numel():
        dense[edge_index[1].to(torch.long), edge_index[0].to(torch.long)] = edge_weight.to(torch.float32)
    return dense
