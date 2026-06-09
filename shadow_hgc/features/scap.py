from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class SourceSCAP:
    counts: torch.Tensor
    totals: torch.Tensor
    active_source_nodes: torch.Tensor | None
    num_source_nodes: int | None = None

    @property
    def affinity(self) -> torch.Tensor:
        return self.counts / self.totals.clamp_min(1.0).unsqueeze(1)

    def affinity_for(self, source_ids: torch.Tensor) -> torch.Tensor:
        source_ids = source_ids.to(device=self.counts.device, dtype=torch.long)
        if self.active_source_nodes is None or int(self.counts.shape[0]) == int(self.num_source_nodes or self.counts.shape[0]):
            return self.affinity[source_ids]
        active = self.active_source_nodes.to(device=self.counts.device, dtype=torch.long)
        if active.numel() == 0:
            return torch.zeros(source_ids.numel(), self.counts.shape[1], dtype=self.counts.dtype, device=self.counts.device)
        pos = torch.searchsorted(active, source_ids)
        valid = (pos < active.numel()) & (active[pos.clamp_max(max(0, active.numel() - 1))] == source_ids)
        out = torch.zeros(source_ids.numel(), self.counts.shape[1], dtype=self.counts.dtype, device=self.counts.device)
        if bool(valid.any()):
            out[valid] = self.affinity[pos[valid]]
        return out

    def totals_for(self, source_ids: torch.Tensor) -> torch.Tensor:
        source_ids = source_ids.to(device=self.totals.device, dtype=torch.long)
        if self.active_source_nodes is None or int(self.totals.shape[0]) == int(self.num_source_nodes or self.totals.shape[0]):
            return self.totals[source_ids]
        active = self.active_source_nodes.to(device=self.totals.device, dtype=torch.long)
        if active.numel() == 0:
            return torch.zeros(source_ids.numel(), dtype=self.totals.dtype, device=self.totals.device)
        pos = torch.searchsorted(active, source_ids)
        valid = (pos < active.numel()) & (active[pos.clamp_max(max(0, active.numel() - 1))] == source_ids)
        out = torch.zeros(source_ids.numel(), dtype=self.totals.dtype, device=self.totals.device)
        if bool(valid.any()):
            out[valid] = self.totals[pos[valid]]
        return out


@dataclass(frozen=True)
class SparseTopKSCAP:
    class_ids: torch.Tensor
    values: torch.Tensor
    num_rows: int
    num_classes: int
    metadata: dict


def _target_lookup(num_nodes: int, target_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    lookup = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
    rows = target_rows.to(device=device, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long, device=device)
    return lookup


def target_target_scap_dense(
    *,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    target_rows: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    device = edge_index.device
    target_rows = target_rows.to(device=device, dtype=torch.long)
    out = torch.zeros(target_rows.numel(), int(num_classes), dtype=torch.float64, device=device)
    if edge_index.numel() == 0 or target_rows.numel() == 0:
        return out
    if alpha is not None:
        rel_alpha = alpha.to(device=device, dtype=torch.float64)
    else:
        dst_for_norm = edge_index[1].to(device=device, dtype=torch.long)
        raw = torch.ones(edge_index.shape[1], dtype=torch.float64, device=device)
        denom = torch.zeros(int(num_nodes), dtype=torch.float64, device=device)
        denom.index_add_(0, dst_for_norm, raw)
        rel_alpha = raw / denom[dst_for_norm].clamp_min(1e-12)
    labels = labels.to(device=device, dtype=torch.long)
    train_mask = train_mask.to(device=device, dtype=torch.bool)
    lookup = _target_lookup(num_nodes, target_rows, device)
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    local_dst = lookup[dst]
    mask = (local_dst >= 0) & train_mask[src] & (labels[src] >= 0) & (labels[src] < int(num_classes))
    if bool(mask.any()):
        flat = local_dst[mask] * int(num_classes) + labels[src[mask]]
        out.view(-1).index_add_(0, flat, rel_alpha[mask])
    return out


def target_target_scap_streaming(
    *,
    edge_chunks: Iterable[torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    chunks = [chunk for chunk in edge_chunks if chunk.numel() > 0]
    if not chunks:
        return torch.zeros(target_rows.numel(), int(num_classes), dtype=torch.float64)
    edge_index = torch.cat(chunks, dim=1)
    return target_target_scap_dense(
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        num_nodes=num_nodes,
        num_classes=num_classes,
        target_rows=target_rows,
    )


def source_class_affinity(
    *,
    source_to_target_edges: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_source_nodes: int,
    num_classes: int,
    active_source_nodes: torch.Tensor | None = None,
) -> SourceSCAP:
    device = source_to_target_edges.device
    if active_source_nodes is not None:
        active_source_nodes = torch.unique(active_source_nodes.to(device=device, dtype=torch.long)).sort().values
        counts = torch.zeros(active_source_nodes.numel(), int(num_classes), dtype=torch.float32, device=device)
        totals = torch.zeros(active_source_nodes.numel(), dtype=torch.float32, device=device)
    else:
        counts = torch.zeros(int(num_source_nodes), int(num_classes), dtype=torch.float32, device=device)
        totals = torch.zeros(int(num_source_nodes), dtype=torch.float32, device=device)
    if source_to_target_edges.numel() == 0:
        return SourceSCAP(counts, totals, active_source_nodes, int(num_source_nodes))
    src = source_to_target_edges[0].to(device=device, dtype=torch.long)
    dst = source_to_target_edges[1].to(device=device, dtype=torch.long)
    labels = labels.to(device=device, dtype=torch.long)
    train_mask = train_mask.to(device=device, dtype=torch.bool)
    mask = train_mask[dst] & (labels[dst] >= 0) & (labels[dst] < int(num_classes))
    if active_source_nodes is not None:
        if active_source_nodes.numel() == 0:
            in_active = torch.zeros_like(mask)
        else:
            pos = torch.searchsorted(active_source_nodes, src)
            in_active = (pos < active_source_nodes.numel()) & (active_source_nodes[pos.clamp_max(active_source_nodes.numel() - 1)] == src)
        mask = mask & in_active
    if bool(mask.any()):
        local_src = src[mask]
        if active_source_nodes is not None:
            local_src = torch.searchsorted(active_source_nodes, local_src)
        flat = local_src * int(num_classes) + labels[dst[mask]]
        counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        totals.index_add_(0, local_src, torch.ones_like(local_src, dtype=torch.float32))
    return SourceSCAP(counts, totals, active_source_nodes, int(num_source_nodes))


def non_target_source_scap(
    *,
    edge_index_source_to_target: torch.Tensor,
    source_affinity: SourceSCAP,
    num_target_nodes: int,
    target_rows: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    device = edge_index_source_to_target.device
    target_rows = target_rows.to(device=device, dtype=torch.long)
    num_classes = int(source_affinity.counts.shape[1])
    out = torch.zeros(target_rows.numel(), num_classes, dtype=torch.float32, device=device)
    if edge_index_source_to_target.numel() == 0 or target_rows.numel() == 0:
        return out, {"active_source_count": 0, "source_fallback_rate": 0.0}
    rel_alpha = alpha.to(device=device, dtype=torch.float32) if alpha is not None else destination_row_normalize(edge_index_source_to_target, int(num_target_nodes)).to(device=device, dtype=torch.float32)
    lookup = _target_lookup(num_target_nodes, target_rows, device)
    src = edge_index_source_to_target[0].to(device=device, dtype=torch.long)
    dst = edge_index_source_to_target[1].to(device=device, dtype=torch.long)
    local_dst = lookup[dst]
    mask = local_dst >= 0
    fallback_edges = torch.zeros((), dtype=torch.long, device=device)
    if bool(mask.any()):
        affinity = source_affinity.affinity_for(src[mask]).to(device=device, dtype=torch.float32)
        weighted = affinity * rel_alpha[mask].unsqueeze(1)
        out.index_add_(0, local_dst[mask], weighted)
        fallback_edges = (source_affinity.totals_for(src[mask]).to(device=device) <= 0).sum()
    active_count = int((source_affinity.totals > 0).sum().item())
    denom = int(mask.sum().item())
    diagnostics = {
        "active_source_count": int(source_affinity.active_source_nodes.numel()) if source_affinity.active_source_nodes is not None else active_count,
        "source_fallback_rate": float(fallback_edges.item() / max(1, denom)),
    }
    return out, diagnostics


def prior_center_scap(
    block: torch.Tensor,
    *,
    train_labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, dict]:
    labels = train_labels.to(torch.long)
    valid = labels[(labels >= 0) & (labels < int(num_classes))]
    counts = torch.bincount(valid, minlength=int(num_classes)).to(block.device, torch.float64)
    prior = counts / counts.sum().clamp_min(1.0)
    mass = block.to(torch.float64).sum(dim=1, keepdim=True)
    centered = block.to(torch.float64) - mass * prior.unsqueeze(0)
    return centered.to(dtype=block.dtype), {
        "prior_centering": True,
        "train_class_prior": [float(value) for value in prior.detach().cpu().tolist()],
    }


def transform_scap_block(
    block: torch.Tensor,
    *,
    log1p: bool = True,
    l2_normalize: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    out = block.to(torch.float32)
    if log1p:
        out = torch.sign(out) * torch.log1p(out.abs())
    if l2_normalize:
        out = out / torch.linalg.norm(out, dim=1, keepdim=True).clamp_min(float(eps))
    return out


def sparse_topk_from_dense(dense: torch.Tensor, *, topk: int) -> SparseTopKSCAP:
    topk = min(int(topk), int(dense.shape[1]))
    values, class_ids = torch.topk(dense.to(torch.float32), k=topk, dim=1)
    return SparseTopKSCAP(
        class_ids=class_ids.to(torch.int32),
        values=values.to(torch.float32),
        num_rows=int(dense.shape[0]),
        num_classes=int(dense.shape[1]),
        metadata={
            "block_type": "scap",
            "dense_or_sparse": "sparse_topk",
            "topk": int(topk),
            "uses_train_labels_only": True,
            "uses_validation_labels": False,
            "uses_test_labels": False,
        },
    )


def dense_from_sparse_topk(sparse: SparseTopKSCAP, *, num_classes: int | None = None) -> torch.Tensor:
    classes = int(num_classes if num_classes is not None else sparse.num_classes)
    dense = torch.zeros(int(sparse.num_rows), classes, dtype=torch.float32, device=sparse.values.device)
    rows = torch.arange(int(sparse.num_rows), device=sparse.values.device).unsqueeze(1).expand_as(sparse.class_ids)
    dense[rows, sparse.class_ids.to(torch.long)] = sparse.values.to(torch.float32)
    return dense


def apply_hub_clipping(
    edge_index: torch.Tensor,
    *,
    hub_cap: int,
    policy: str = "clip",
) -> tuple[torch.Tensor, dict]:
    if policy not in {"clip", "sample", "topk_targets"}:
        raise ValueError("hub policy must be clip, sample, or topk_targets")
    if int(hub_cap) <= 0 or edge_index.numel() == 0:
        return edge_index[:, :0], {
            "num_hub_clipped_sources": 0,
            "fraction_edges_clipped": 1.0 if edge_index.numel() else 0.0,
            "max_source_degree_before_clip": 0,
            "max_source_degree_after_clip": 0,
        }
    src = edge_index[0].to(torch.long)
    keep = torch.zeros(src.numel(), dtype=torch.bool, device=edge_index.device)
    counts: dict[int, int] = {}
    degree: dict[int, int] = {}
    for pos, source in enumerate(src.detach().cpu().tolist()):
        degree[source] = degree.get(source, 0) + 1
        current = counts.get(source, 0)
        if current < int(hub_cap):
            keep[pos] = True
            counts[source] = current + 1
    clipped = edge_index[:, keep]
    before_max = max(degree.values()) if degree else 0
    after_max = max(counts.values()) if counts else 0
    clipped_sources = sum(1 for value in degree.values() if value > int(hub_cap))
    removed = int(edge_index.shape[1]) - int(clipped.shape[1])
    return clipped, {
        "num_hub_clipped_sources": int(clipped_sources),
        "fraction_edges_clipped": float(removed / max(1, int(edge_index.shape[1]))),
        "max_source_degree_before_clip": int(before_max),
        "max_source_degree_after_clip": int(after_max),
        "hub_policy": policy,
        "hub_cap": int(hub_cap),
    }
