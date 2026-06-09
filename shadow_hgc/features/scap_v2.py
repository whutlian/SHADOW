from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.features.scap_sparse import SCAPTopK, scap_topk_from_dense
from shadow_hgc.features.scap_stats import scap_row_stats


@dataclass(frozen=True)
class SCAPV2Block:
    dense: torch.Tensor | None
    sparse: SCAPTopK | None
    support_count: torch.Tensor
    row_entropy: torch.Tensor
    max_affinity: torch.Tensor
    missingness: torch.Tensor
    diagnostics: dict


def _target_lookup(num_nodes: int, target_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    lookup = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
    rows = target_rows.to(device=device, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long, device=device)
    return lookup


def prior_center_dense(
    dense: torch.Tensor,
    *,
    train_labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, dict]:
    labels = train_labels.to(torch.long)
    valid = labels[(labels >= 0) & (labels < int(num_classes))]
    counts = torch.bincount(valid, minlength=int(num_classes)).to(dense.device, torch.float64)
    prior = counts / counts.sum().clamp_min(1.0)
    mass = dense.to(torch.float64).sum(dim=1, keepdim=True)
    centered = dense.to(torch.float64) - mass * prior.unsqueeze(0)
    return centered.to(dense.dtype), {
        "prior_centering": True,
        "class_prior": [float(value) for value in prior.detach().cpu().tolist()],
    }


def compute_target_target_scap_v2(
    *,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    target_rows: torch.Tensor,
    prior_center: bool = True,
    top_k: int = 8,
    sparse: bool = False,
) -> SCAPV2Block:
    device = edge_index.device
    rows = target_rows.to(device=device, dtype=torch.long)
    dense = torch.zeros(rows.numel(), int(num_classes), dtype=torch.float64, device=device)
    if edge_index.numel() > 0 and rows.numel() > 0:
        edge_index = edge_index.to(device=device, dtype=torch.long)
        dst = edge_index[1]
        raw = torch.ones(edge_index.shape[1], dtype=torch.float64, device=device)
        denom = torch.zeros(int(num_nodes), dtype=torch.float64, device=device)
        denom.index_add_(0, dst, raw)
        alpha = raw / denom[dst].clamp_min(1e-12)
        lookup = _target_lookup(num_nodes, rows, device)
        src = edge_index[0]
        local_dst = lookup[dst]
        labels = labels.to(device=device, dtype=torch.long)
        train_mask = train_mask.to(device=device, dtype=torch.bool)
        mask = (local_dst >= 0) & train_mask[src] & (labels[src] >= 0) & (labels[src] < int(num_classes))
        if bool(mask.any()):
            flat = local_dst[mask] * int(num_classes) + labels[src[mask]]
            dense.view(-1).index_add_(0, flat, alpha[mask])
    diagnostics = {
        "uses_train_labels_only": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "normalization": "destination_row",
        "scap_topk": int(top_k),
        "dense_or_sparse": "sparse_topk" if sparse else "dense",
        "uses_dense_p2": False,
    }
    if prior_center:
        dense, prior_diag = prior_center_dense(dense.to(torch.float32), train_labels=labels[train_mask], num_classes=num_classes)
        diagnostics.update(prior_diag)
    stats = scap_row_stats(dense.to(torch.float32))
    diagnostics["scap_support_count_stats"] = stats["support_count_stats"]
    diagnostics["scap_entropy_stats"] = stats["entropy_stats"]
    if sparse:
        sparse_block = scap_topk_from_dense(dense.to(torch.float32), top_k=top_k)
        diagnostics["scap_cache_bytes"] = int(sparse_block.values.numel() * sparse_block.values.element_size() + sparse_block.class_ids.numel() * sparse_block.class_ids.element_size())
        return SCAPV2Block(None, sparse_block, stats["support_count"], stats["row_entropy"], stats["max_affinity"], stats["missingness"], diagnostics)
    diagnostics["scap_cache_bytes"] = int(dense.numel() * dense.element_size())
    return SCAPV2Block(dense, None, stats["support_count"], stats["row_entropy"], stats["max_affinity"], stats["missingness"], diagnostics)


def clip_source_hubs(edge_index: torch.Tensor, *, hub_cap: int) -> tuple[torch.Tensor, dict]:
    if int(hub_cap) <= 0 or edge_index.numel() == 0:
        return edge_index[:, :0], {"hub_cap": int(hub_cap), "num_clipped_hubs": 0, "fraction_edges_clipped": 1.0 if edge_index.numel() else 0.0}
    src = edge_index[0].to(torch.long)
    keep = torch.zeros(src.numel(), dtype=torch.bool, device=edge_index.device)
    seen: dict[int, int] = {}
    degree: dict[int, int] = {}
    for pos, value in enumerate(src.detach().cpu().tolist()):
        degree[value] = degree.get(value, 0) + 1
        count = seen.get(value, 0)
        if count < int(hub_cap):
            keep[pos] = True
            seen[value] = count + 1
    clipped = edge_index[:, keep]
    return clipped, {
        "hub_cap": int(hub_cap),
        "num_clipped_hubs": int(sum(1 for value in degree.values() if value > int(hub_cap))),
        "fraction_edges_clipped": float((int(edge_index.shape[1]) - int(clipped.shape[1])) / max(1, int(edge_index.shape[1]))),
        "max_source_degree_before_clip": int(max(degree.values()) if degree else 0),
        "max_source_degree_after_clip": int(max(seen.values()) if seen else 0),
    }


def validate_scap_v2_config(config: dict) -> dict:
    invalid: list[str] = []
    if bool(config.get("uses_dense_p2", False)):
        invalid.append("uses_dense_p2")
    if bool(config.get("uses_high_dim_diffusion", False)):
        invalid.append("uses_high_dim_diffusion")
    return {"valid": len(invalid) == 0, "invalid_reasons": invalid}
