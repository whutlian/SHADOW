from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass
class SourceLabelAffinity:
    counts: torch.Tensor
    totals: torch.Tensor
    active_source_nodes: torch.Tensor | None = None

    @property
    def affinity(self) -> torch.Tensor:
        return self.counts / self.totals.clamp_min(1.0).unsqueeze(1)


@dataclass
class LabelAffinityBlockStats:
    mean: torch.Tensor
    std: torch.Tensor
    l1_norm_mean: float
    l2_norm_mean: float
    zero_row_ratio: float

    def to_json(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "l1_norm_mean": self.l1_norm_mean,
            "l2_norm_mean": self.l2_norm_mean,
            "zero_row_ratio": self.zero_row_ratio,
        }


def _target_node_rows(num_nodes: int, target_nodes: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    if target_nodes is None:
        nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
        return nodes, None
    nodes = target_nodes.to(device=device, dtype=torch.long)
    lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    lookup[nodes] = torch.arange(nodes.numel(), dtype=torch.long, device=device)
    return nodes, lookup


def compute_target_target_label_affinity(
    edge_index: torch.Tensor,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    alpha: torch.Tensor | None = None,
    exclude_self: bool = True,
    dtype: torch.dtype = torch.float32,
    target_nodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return train-label-only target-side affinity for a target-target relation."""

    device = edge_index.device
    rows, target_lookup = _target_node_rows(num_nodes, target_nodes, device)
    out = torch.zeros(rows.numel(), num_classes, dtype=dtype, device=device)
    if edge_index.numel() == 0 or rows.numel() == 0:
        return out

    train_mask = train_target_mask.to(device=device, dtype=torch.bool)
    labels = train_labels.to(device=device, dtype=torch.long)
    rel_alpha = alpha.to(device=device, dtype=dtype) if alpha is not None else destination_row_normalize(edge_index, num_nodes).to(dtype)

    src = edge_index[0].to(device=device)
    dst = edge_index[1].to(device=device)
    mask = train_mask[src] & (labels[src] >= 0) & (labels[src] < num_classes)
    if exclude_self:
        mask = mask & (src != dst)
    if target_lookup is not None:
        local_dst = target_lookup[dst]
        mask = mask & (local_dst >= 0)
    else:
        local_dst = dst
    if not bool(mask.any()):
        return out

    flat = local_dst[mask] * int(num_classes) + labels[src[mask]]
    out.view(-1).index_add_(0, flat.to(torch.long), rel_alpha[mask].to(dtype))
    return out


def compute_source_label_counts(
    source_to_target_edges: torch.Tensor,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_source_nodes: int,
    num_classes: int,
    active_source_nodes: torch.Tensor | None = None,
) -> SourceLabelAffinity:
    """Build source train-label histograms using train target labels only."""

    device = source_to_target_edges.device
    counts = torch.zeros(num_source_nodes, num_classes, dtype=torch.float32, device=device)
    totals = torch.zeros(num_source_nodes, dtype=torch.float32, device=device)
    if source_to_target_edges.numel() == 0:
        return SourceLabelAffinity(counts=counts, totals=totals, active_source_nodes=active_source_nodes)

    src = source_to_target_edges[0].to(device=device)
    dst = source_to_target_edges[1].to(device=device)
    train_mask = train_target_mask.to(device=device, dtype=torch.bool)
    labels = train_labels.to(device=device, dtype=torch.long)
    mask = train_mask[dst] & (labels[dst] >= 0) & (labels[dst] < num_classes)
    if active_source_nodes is not None:
        active = torch.zeros(num_source_nodes, dtype=torch.bool, device=device)
        active[active_source_nodes.to(device=device, dtype=torch.long)] = True
        mask = mask & active[src]
    if bool(mask.any()):
        flat = src[mask] * int(num_classes) + labels[dst[mask]]
        counts.view(-1).index_add_(0, flat.to(torch.long), torch.ones_like(flat, dtype=counts.dtype))
        totals.index_add_(0, src[mask], torch.ones_like(src[mask], dtype=totals.dtype))
    return SourceLabelAffinity(counts=counts, totals=totals, active_source_nodes=active_source_nodes)


def aggregate_non_target_label_affinity(
    edge_index_source_to_target: torch.Tensor,
    source_affinity: SourceLabelAffinity,
    target_nodes: torch.Tensor,
    target_train_labels: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    leave_one_out_for_train: bool = True,
) -> torch.Tensor:
    """Aggregate source train-label affinities to requested target rows."""

    device = edge_index_source_to_target.device
    target_nodes = target_nodes.to(device=device, dtype=torch.long)
    num_classes = int(source_affinity.counts.shape[1])
    out = torch.zeros(target_nodes.numel(), num_classes, dtype=source_affinity.counts.dtype, device=device)
    if edge_index_source_to_target.numel() == 0 or target_nodes.numel() == 0:
        return out

    num_targets = int(max(int(edge_index_source_to_target[1].max().item()) + 1, int(target_nodes.max().item()) + 1))
    lookup = torch.full((num_targets,), -1, dtype=torch.long, device=device)
    lookup[target_nodes] = torch.arange(target_nodes.numel(), dtype=torch.long, device=device)
    rel_alpha = (
        alpha.to(device=device, dtype=out.dtype)
        if alpha is not None
        else destination_row_normalize(edge_index_source_to_target, num_targets).to(out.dtype)
    )
    src = edge_index_source_to_target[0].to(device=device)
    dst = edge_index_source_to_target[1].to(device=device)
    local_dst = lookup[dst]
    mask = local_dst >= 0
    if not bool(mask.any()):
        return out

    counts = source_affinity.counts.to(device=device, dtype=out.dtype)
    totals = source_affinity.totals.to(device=device, dtype=out.dtype)
    edge_counts = counts[src[mask]].clone()
    edge_totals = totals[src[mask]].clone()

    if leave_one_out_for_train and target_train_labels is not None:
        labels = target_train_labels.to(device=device, dtype=torch.long)
        edge_dst = dst[mask]
        train_edge = (labels[edge_dst] >= 0) & (labels[edge_dst] < num_classes)
        if bool(train_edge.any()):
            row_idx = torch.nonzero(train_edge, as_tuple=False).flatten()
            cls = labels[edge_dst[train_edge]]
            edge_counts[row_idx, cls] -= 1.0
            edge_totals[row_idx] -= 1.0
            edge_counts.clamp_(min=0.0)
            edge_totals.clamp_(min=0.0)

    aff = edge_counts / edge_totals.clamp_min(1.0).unsqueeze(1)
    weighted = aff * rel_alpha[mask].unsqueeze(1)
    out.index_add_(0, local_dst[mask], weighted)
    return out


def normalize_label_affinity_block(
    block: torch.Tensor,
    *,
    mode: str = "row_l1",
    fit_rows: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, LabelAffinityBlockStats]:
    if mode not in {"none", "row_l1", "standardize", "standardize_l2"}:
        raise ValueError("mode must be none, row_l1, standardize, or standardize_l2")
    block = block.to(torch.float32)
    fit = block if fit_rows is None else block[fit_rows.to(torch.long)]
    mean = fit.mean(dim=0) if fit.numel() else torch.zeros(block.shape[1], dtype=block.dtype, device=block.device)
    std = fit.std(dim=0, unbiased=False).clamp_min(eps) if fit.numel() else torch.ones(block.shape[1], dtype=block.dtype, device=block.device)
    out = block.clone()
    if mode == "row_l1":
        out = out / out.abs().sum(dim=1, keepdim=True).clamp_min(eps)
    elif mode == "standardize":
        out = (out - mean) / std
    elif mode == "standardize_l2":
        out = (out - mean) / std
        out = out / torch.linalg.norm(out, dim=1, keepdim=True).clamp_min(eps)
    l1 = block.abs().sum(dim=1)
    l2 = torch.linalg.norm(block, dim=1)
    stats = LabelAffinityBlockStats(
        mean=mean.detach().clone(),
        std=std.detach().clone(),
        l1_norm_mean=float(l1.mean().item()) if l1.numel() else 0.0,
        l2_norm_mean=float(l2.mean().item()) if l2.numel() else 0.0,
        zero_row_ratio=(int((l1 <= eps).sum().item()) / int(l1.numel())) if l1.numel() else 0.0,
    )
    return out, stats
