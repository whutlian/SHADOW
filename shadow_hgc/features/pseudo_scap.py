from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class PseudoLabelResult:
    pseudo: torch.Tensor
    weights: torch.Tensor
    confidence: torch.Tensor
    diagnostics: dict


@dataclass(frozen=True)
class PseudoTopK:
    indices: torch.Tensor
    values: torch.Tensor
    support: torch.Tensor | None
    entropy: torch.Tensor | None
    max_confidence: torch.Tensor | None
    num_rows: int
    num_classes: int
    metadata: dict


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.to(torch.long), num_classes=int(num_classes)).to(torch.float32)


def build_pseudo_labels(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    threshold: float,
    pseudo_weight: float,
    temperature: float = 1.0,
) -> PseudoLabelResult:
    z = logits.to(torch.float32)
    labels = labels.to(device=z.device, dtype=torch.long)
    train_idx = train_idx.to(device=z.device, dtype=torch.long)
    probs = torch.softmax(z / max(float(temperature), 1e-12), dim=1)
    confidence = probs.max(dim=1).values
    weights = torch.zeros(int(z.shape[0]), dtype=torch.float32, device=z.device)
    train_mask = torch.zeros(int(z.shape[0]), dtype=torch.bool, device=z.device)
    if train_idx.numel() > 0:
        train_mask[train_idx] = True
        probs[train_idx] = _one_hot(labels[train_idx], int(z.shape[1]))
        weights[train_idx] = 1.0
    nontrain_mask = ~train_mask
    keep = nontrain_mask & (confidence >= float(threshold))
    weights[keep] = float(pseudo_weight)
    nontrain_count = int(keep.sum().item())
    nontrain_total = int(nontrain_mask.sum().item())
    diagnostics = {
        "threshold": float(threshold),
        "pseudo_weight": float(pseudo_weight),
        "temperature": float(temperature),
        "train_override_count": int(train_idx.numel()),
        "nontrain_used_count": nontrain_count,
        "pseudo_coverage": 0.0 if nontrain_total == 0 else float(nontrain_count / nontrain_total),
        "mean_confidence": float(confidence.mean().item()) if confidence.numel() else 0.0,
        "median_confidence": float(confidence.median().item()) if confidence.numel() else 0.0,
        "uses_train_labels": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
    }
    return PseudoLabelResult(pseudo=probs, weights=weights, confidence=confidence, diagnostics=diagnostics)


def dense_to_topk_sparse(
    values: torch.Tensor,
    *,
    topk: int,
    support: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    max_confidence: torch.Tensor | None = None,
) -> PseudoTopK:
    dense = values.to(torch.float32)
    k = min(int(topk), int(dense.shape[1]))
    top_values, top_indices = torch.topk(dense, k=k, dim=1)
    return PseudoTopK(
        indices=top_indices.to(torch.int64),
        values=top_values,
        support=None if support is None else support.to(torch.float32),
        entropy=None if entropy is None else entropy.to(torch.float32),
        max_confidence=None if max_confidence is None else max_confidence.to(torch.float32),
        num_rows=int(dense.shape[0]),
        num_classes=int(dense.shape[1]),
        metadata={"dense_or_sparse": "sparse_topk", "topk_classes": int(k), "propagates_features": False},
    )


def dense_from_topk_sparse(sparse: PseudoTopK) -> torch.Tensor:
    dense = torch.zeros(int(sparse.num_rows), int(sparse.num_classes), dtype=torch.float32, device=sparse.values.device)
    rows = torch.arange(int(sparse.num_rows), device=sparse.values.device).unsqueeze(1).expand_as(sparse.indices)
    dense[rows, sparse.indices.to(torch.long)] = sparse.values.to(torch.float32)
    return dense


def train_class_prior(labels: torch.Tensor, train_idx: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    if train_idx.numel() == 0:
        return torch.full((int(num_classes),), 1.0 / max(1, int(num_classes)), dtype=torch.float32, device=labels.device)
    y = labels.to(torch.long)[train_idx.to(torch.long)]
    counts = torch.bincount(y, minlength=int(num_classes)).to(torch.float32)
    return counts / counts.sum().clamp_min(1.0)


def apply_prior_centering(affinity: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    return affinity.to(torch.float32) - prior.to(device=affinity.device, dtype=torch.float32).view(1, -1)


def target_target_pseudo_scap(
    *,
    edge_index: torch.Tensor,
    pseudo: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
    raw_edge_weight: torch.Tensor | None = None,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, dict]:
    affinity = torch.zeros(int(num_nodes), int(pseudo.shape[1]), dtype=torch.float32, device=pseudo.device)
    support = torch.zeros(int(num_nodes), dtype=torch.float32, device=pseudo.device)
    if edge_index.numel() == 0:
        return affinity, {"normalization": "destination_row", "edge_count": 0, "propagates_features": False}
    edge_index = edge_index.to(device=pseudo.device, dtype=torch.long)
    alpha = destination_row_normalize(edge_index, int(num_nodes), raw_edge_weight=raw_edge_weight).to(device=pseudo.device, dtype=torch.float32)
    weighted = pseudo.to(torch.float32) * weights.to(torch.float32).view(-1, 1)
    for start in range(0, int(edge_index.shape[1]), int(chunk_size)):
        end = min(int(edge_index.shape[1]), start + int(chunk_size))
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]
        coeff = alpha[start:end].unsqueeze(1)
        affinity.index_add_(0, dst, weighted[src] * coeff)
        support.index_add_(0, dst, weights[src].to(torch.float32) * alpha[start:end])
    diagnostics = {
        "normalization": "destination_row",
        "edge_count": int(edge_index.shape[1]),
        "support_nonzero_count": int((support > 0).sum().item()),
        "propagates_features": False,
    }
    return affinity, diagnostics


def source_affinity_from_target_edges(
    *,
    edge_index: torch.Tensor,
    pseudo: torch.Tensor,
    weights: torch.Tensor,
    num_source_nodes: int,
) -> torch.Tensor:
    source = torch.zeros(int(num_source_nodes), int(pseudo.shape[1]), dtype=torch.float32, device=pseudo.device)
    denom = torch.zeros(int(num_source_nodes), dtype=torch.float32, device=pseudo.device)
    if edge_index.numel() == 0:
        return source
    src = edge_index[0].to(device=pseudo.device, dtype=torch.long)
    target = edge_index[1].to(device=pseudo.device, dtype=torch.long)
    weighted = pseudo[target].to(torch.float32) * weights[target].to(torch.float32).view(-1, 1)
    source.index_add_(0, src, weighted)
    denom.index_add_(0, src, weights[target].to(torch.float32))
    return source / denom.clamp_min(1e-12).view(-1, 1)
