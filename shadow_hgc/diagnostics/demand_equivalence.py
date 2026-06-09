from __future__ import annotations

from typing import Any

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


def compute_destination_row_feature_demand(
    *,
    edge_index: torch.Tensor,
    source_features: torch.Tensor,
    num_target_nodes: int,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if source_features.ndim != 2:
        raise ValueError("source_features must have shape [num_source_nodes, feature_dim]")
    rows = target_rows.to(dtype=torch.long)
    out = torch.zeros(rows.numel(), source_features.shape[1], dtype=torch.float32)
    if edge_index.numel() == 0 or rows.numel() == 0:
        return out
    alpha = destination_row_normalize(edge_index.to(torch.long), int(num_target_nodes)).to(torch.float32)
    lookup = torch.full((int(num_target_nodes),), -1, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long)
    dst_local = lookup[edge_index[1].to(torch.long)]
    mask = dst_local >= 0
    if bool(mask.any()):
        messages = source_features.to(torch.float32)[edge_index[0, mask].to(torch.long)] * alpha[mask].unsqueeze(1)
        out.index_add_(0, dst_local[mask], messages)
    return out


def compare_relation_demand_blocks(
    dataset: str,
    relation_name: str,
    demand_a: torch.Tensor,
    demand_b: torch.Tensor,
    train_target_ids: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    *,
    source_type: str = "",
    destination_type: str = "",
    edge_direction_checked: bool = False,
    alpha_normalization_checked: bool = False,
) -> dict[str, Any]:
    a = demand_a.detach().to(torch.float32)
    b = demand_b.detach().to(torch.float32)
    if a.shape != b.shape:
        row_l2 = torch.full((max(a.shape[0], b.shape[0]),), float("inf"))
        cosine = torch.full_like(row_l2, -1.0)
        allclose_fraction = 0.0
    else:
        diff = a - b
        row_l2 = torch.linalg.vector_norm(diff, dim=1)
        denom = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
        raw_cosine = (a * b).sum(dim=1) / denom.clamp_min(1e-12)
        both_zero = (torch.linalg.vector_norm(a, dim=1) <= 1e-12) & (torch.linalg.vector_norm(b, dim=1) <= 1e-12)
        cosine = torch.where(both_zero, torch.ones_like(raw_cosine), raw_cosine)
        row_close = torch.isclose(a, b, atol=float(atol), rtol=float(rtol)).all(dim=1)
        allclose_fraction = float(row_close.to(torch.float32).mean().item()) if row_close.numel() else 1.0
    finite_l2 = row_l2[torch.isfinite(row_l2)]
    finite_cosine = cosine[torch.isfinite(cosine)]
    return {
        "dataset": dataset,
        "relation_name": relation_name,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "num_train_target_ids": int(train_target_ids.numel()),
        "row_l2_mean": float(finite_l2.mean().item()) if finite_l2.numel() else float("inf"),
        "row_l2_median": float(finite_l2.median().item()) if finite_l2.numel() else float("inf"),
        "row_l2_max": float(finite_l2.max().item()) if finite_l2.numel() else float("inf"),
        "cosine_mean": float(finite_cosine.mean().item()) if finite_cosine.numel() else -1.0,
        "cosine_min": float(finite_cosine.min().item()) if finite_cosine.numel() else -1.0,
        "allclose_fraction": allclose_fraction,
        "nan_count_a": int(torch.isnan(a).sum().item()),
        "nan_count_b": int(torch.isnan(b).sum().item()),
        "source_type": source_type,
        "destination_type": destination_type,
        "edge_direction_checked": bool(edge_direction_checked),
        "alpha_normalization_checked": bool(alpha_normalization_checked),
    }
