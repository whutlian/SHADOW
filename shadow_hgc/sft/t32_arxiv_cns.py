from __future__ import annotations

from typing import Any

import torch


def cns_grid_plan_v2(
    *,
    graph_directions: list[str],
    correction_alphas: list[float],
    smoothing_alphas: list[float],
    correction_steps: list[int],
    smoothing_steps: list[int],
    autoscale: list[str],
    normalization_modes: list[str],
    self_loop_modes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in graph_directions:
        for norm in normalization_modes:
            for self_loop in self_loop_modes:
                for auto in autoscale:
                    for ca in correction_alphas:
                        for sa in smoothing_alphas:
                            for cs in correction_steps:
                                for ss in smoothing_steps:
                                    rows.append(
                                        {
                                            "graph_direction": str(direction),
                                            "normalization_mode": str(norm),
                                            "self_loop_mode": str(self_loop),
                                            "autoscale": str(auto),
                                            "correction_alpha": float(ca),
                                            "smoothing_alpha": float(sa),
                                            "correction_steps": int(cs),
                                            "smoothing_steps": int(ss),
                                        }
                                    )
    return rows


def cns_failure_reason(*, cns_accuracy: float | str, base_predictor: str) -> str:
    del base_predictor
    try:
        acc = float(cns_accuracy)
    except (TypeError, ValueError):
        return "missing_cns_accuracy"
    if acc < 0.65:
        return "cns_pipeline_mismatch_or_weak_base"
    if acc < 0.715:
        return "below_arxiv_safe_teacher_gate"
    return ""


def transform_arxiv_edge_index(edge_index: torch.Tensor, *, graph_direction: str, self_loop_mode: str = "none") -> torch.Tensor:
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    direction = str(graph_direction)
    if direction == "cite_ref":
        out = edge_index.to(torch.long)
    elif direction == "cited_by":
        out = edge_index.flip(0).to(torch.long)
    elif direction == "undirected_sym":
        out = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(torch.long)
    else:
        raise ValueError(f"unsupported graph_direction: {graph_direction}")
    if str(self_loop_mode) == "target_all":
        num_nodes = int(out.max().item()) + 1 if out.numel() else 0
        loop = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
        out = torch.cat([out, loop], dim=1)
    elif str(self_loop_mode) != "none":
        raise ValueError(f"unsupported self_loop_mode: {self_loop_mode}")
    return out.contiguous().clone()
