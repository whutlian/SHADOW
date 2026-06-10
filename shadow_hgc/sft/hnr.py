from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from shadow_hgc.data.edge_stream import EdgeChunk


@dataclass(frozen=True)
class HNRStats:
    target_rows: torch.Tensor
    degree: torch.Tensor
    labeled_support: torch.Tensor
    same_label_support: torch.Tensor
    label_max_count: torch.Tensor
    label_second_count: torch.Tensor
    label_entropy: torch.Tensor
    label_max_affinity: torch.Tensor
    missing_label_ratio: torch.Tensor
    homophily: torch.Tensor
    quality: torch.Tensor
    node_weight: torch.Tensor
    stratum: list[str]
    hnr_cache_bytes: int
    hnr_edge_scans: int


def _as_long_cpu(values: torch.Tensor) -> torch.Tensor:
    return values.detach().to(torch.long).cpu()


def _classwise_robust_z(values: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(values, dtype=torch.float32)
    for cls in torch.unique(labels[train_mask], sorted=True):
        cls_mask = train_mask & (labels == int(cls.item()))
        if not bool(cls_mask.any()):
            continue
        x = values[cls_mask].to(torch.float32)
        med = x.median()
        mad = (x - med).abs().median().clamp_min(1e-6)
        z[cls_mask] = (x - med) / (1.4826 * mad)
    return z.clamp(-8.0, 8.0)


def compute_streaming_hnr_stats(
    *,
    edge_stream_factory: Callable[[], Iterable[EdgeChunk]],
    num_nodes: int,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    target_rows: torch.Tensor,
    num_classes: int | None = None,
    beta: float = 0.5,
    tau: float = 1.0,
    w_min: float = 0.10,
    w_max: float = 2.00,
    eps: float = 1e-12,
) -> HNRStats:
    """Compute directed train-label-only HNR statistics for target destinations.

    The stream convention is fixed: `chunk.src` sends messages to `chunk.dst`.
    Only labels of nodes in `train_rows` are allowed to affect label-support
    statistics. Degree still counts all incoming streamed edges.
    """

    labels = _as_long_cpu(labels)
    train_rows = _as_long_cpu(train_rows)
    target_rows = _as_long_cpu(target_rows)
    if num_classes is None:
        train_labels = labels[train_rows] if train_rows.numel() else labels[:0]
        num_classes = int(train_labels.max().item()) + 1 if train_labels.numel() else 0
    num_classes = max(1, int(num_classes))
    num_targets = int(target_rows.numel())

    target_pos = torch.full((int(num_nodes),), -1, dtype=torch.long)
    target_pos[target_rows] = torch.arange(num_targets, dtype=torch.long)
    train_label_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    train_label_mask[train_rows] = True

    degree = torch.zeros(num_targets, dtype=torch.long)
    hist = torch.zeros((num_targets, num_classes), dtype=torch.long)
    same_label_support = torch.zeros(num_targets, dtype=torch.long)

    scans = 0
    for chunk in edge_stream_factory():
        scans = 1
        src = _as_long_cpu(chunk.src)
        dst = _as_long_cpu(chunk.dst)
        pos = target_pos[dst]
        in_targets = pos >= 0
        if bool(in_targets.any()):
            degree += torch.bincount(pos[in_targets], minlength=num_targets).to(torch.long)
        labeled = in_targets & train_label_mask[src]
        if bool(labeled.any()):
            labeled_pos = pos[labeled]
            src_label = labels[src[labeled]].clamp(0, num_classes - 1)
            flat = labeled_pos * num_classes + src_label
            hist += torch.bincount(flat, minlength=num_targets * num_classes).view(num_targets, num_classes)
            dst_train = train_label_mask[dst[labeled]]
            same = torch.zeros_like(labeled_pos, dtype=torch.bool)
            if bool(dst_train.any()):
                labeled_dst = dst[labeled]
                same[dst_train] = labels[src[labeled][dst_train]] == labels[labeled_dst[dst_train]]
            if bool(same.any()):
                same_label_support += torch.bincount(labeled_pos[same], minlength=num_targets).to(torch.long)

    labeled_support = hist.sum(dim=1)
    sorted_counts = torch.sort(hist, dim=1, descending=True).values
    label_max_count = sorted_counts[:, 0]
    label_second_count = sorted_counts[:, 1] if num_classes > 1 else torch.zeros_like(label_max_count)
    support_f = labeled_support.to(torch.float32)
    probs = hist.to(torch.float32) / support_f.clamp_min(1.0).unsqueeze(1)
    entropy = -(probs.clamp_min(float(eps)).log() * probs).sum(dim=1)
    if num_classes > 1:
        entropy = entropy / torch.log(torch.tensor(float(num_classes)))
    entropy = torch.where(support_f > 0, entropy, torch.zeros_like(entropy))
    homophily = same_label_support.to(torch.float32) / support_f.clamp_min(float(eps))
    homophily = torch.where(support_f > 0, homophily, torch.zeros_like(homophily))
    label_max_affinity = label_max_count.to(torch.float32) / support_f.clamp_min(float(eps))
    label_max_affinity = torch.where(support_f > 0, label_max_affinity, torch.zeros_like(label_max_affinity))
    missing_label_ratio = 1.0 - support_f / degree.to(torch.float32).clamp_min(1.0)
    missing_label_ratio = torch.where(degree > 0, missing_label_ratio, torch.ones_like(missing_label_ratio))
    quality = torch.log1p(support_f) * (1.0 - entropy.clamp(0.0, 1.0))

    target_train_mask = train_label_mask[target_rows]
    target_labels = torch.full((num_targets,), -1, dtype=torch.long)
    if bool(target_train_mask.any()):
        target_labels[target_train_mask] = labels[target_rows[target_train_mask]]
    z_h = _classwise_robust_z(homophily, target_labels, target_train_mask)
    z_q = _classwise_robust_z(quality, target_labels, target_train_mask)
    raw_score = (z_h + float(beta) * z_q) / max(float(tau), 1e-6)
    node_weight = torch.sigmoid(raw_score).clamp(float(w_min), float(w_max))
    stratum = []
    for idx in range(num_targets):
        if not bool(target_train_mask[idx]):
            stratum.append("H0")
        elif float(node_weight[idx].item()) >= 0.62:
            stratum.append("H+")
        elif float(node_weight[idx].item()) <= 0.38:
            stratum.append("H-")
        else:
            stratum.append("H0")

    cache_bytes = int((degree.numel() * 5 + hist.numel()) * 8)
    return HNRStats(
        target_rows=target_rows,
        degree=degree,
        labeled_support=labeled_support,
        same_label_support=same_label_support,
        label_max_count=label_max_count,
        label_second_count=label_second_count,
        label_entropy=entropy.to(torch.float32),
        label_max_affinity=label_max_affinity.to(torch.float32),
        missing_label_ratio=missing_label_ratio.to(torch.float32),
        homophily=homophily.to(torch.float32),
        quality=quality.to(torch.float32),
        node_weight=node_weight.to(torch.float32),
        stratum=stratum,
        hnr_cache_bytes=cache_bytes,
        hnr_edge_scans=scans,
    )
