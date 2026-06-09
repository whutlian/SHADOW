from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation


@dataclass
class SourceAnchorResult:
    anchor_indices: torch.Tensor
    scores: torch.Tensor
    label_purity: torch.Tensor
    coverage: torch.Tensor
    exposed_source_type: str
    exposed_relation: DirectedRelation
    diagnostics: dict


@dataclass
class AnchorResidualResult:
    anchor_message: torch.Tensor
    residual: torch.Tensor
    residual_energy_before: float
    residual_energy_after: float


def _source_train_label_counts(
    edge_index_source_to_target: torch.Tensor,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_source_nodes: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = edge_index_source_to_target.device
    counts = torch.zeros(num_source_nodes, num_classes, dtype=torch.float32, device=device)
    train_totals = torch.zeros(num_source_nodes, dtype=torch.float32, device=device)
    degree = torch.zeros(num_source_nodes, dtype=torch.float32, device=device)
    if edge_index_source_to_target.numel() == 0:
        return counts, train_totals, degree
    src = edge_index_source_to_target[0].to(device=device, dtype=torch.long)
    dst = edge_index_source_to_target[1].to(device=device, dtype=torch.long)
    degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    mask = train_target_mask.to(device=device, dtype=torch.bool)[dst]
    labels = train_labels.to(device=device, dtype=torch.long)
    mask = mask & (labels[dst] >= 0) & (labels[dst] < num_classes)
    if bool(mask.any()):
        flat = src[mask] * int(num_classes) + labels[dst[mask]]
        counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        train_totals.index_add_(0, src[mask], torch.ones_like(src[mask], dtype=torch.float32))
    return counts, train_totals, degree


def select_source_anchors(
    edge_index_source_to_target: torch.Tensor,
    relation: DirectedRelation,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_source_nodes: int,
    num_classes: int,
    max_anchors: int,
    min_anchors: int = 0,
    lambda_purity: float = 1.0,
    lambda_coverage: float = 1.0,
    lambda_degree: float = 0.1,
) -> SourceAnchorResult:
    counts, train_totals, degree = _source_train_label_counts(
        edge_index_source_to_target,
        train_target_mask,
        train_labels,
        num_source_nodes=int(num_source_nodes),
        num_classes=int(num_classes),
    )
    purity = counts.max(dim=1).values / train_totals.clamp_min(1.0)
    coverage = train_totals
    degree_score = torch.log1p(degree.clamp(max=64.0))
    scores = float(lambda_purity) * purity + float(lambda_coverage) * coverage + float(lambda_degree) * degree_score
    budget = min(int(num_source_nodes), max(int(min_anchors), int(max_anchors)))
    if budget <= 0:
        anchors = torch.empty(0, dtype=torch.long, device=edge_index_source_to_target.device)
    else:
        anchors = torch.topk(scores, k=budget).indices.sort().values
    diagnostics = {
        "num_anchors": int(anchors.numel()),
        "anchor_coverage": float(coverage[anchors].sum().item() / coverage.sum().clamp_min(1.0).item()) if anchors.numel() else 0.0,
        "anchor_label_purity_mean": float(purity[anchors].mean().item()) if anchors.numel() else 0.0,
        "uses_train_labels_only": True,
    }
    return SourceAnchorResult(
        anchor_indices=anchors.cpu(),
        scores=scores.detach().cpu(),
        label_purity=purity.detach().cpu(),
        coverage=coverage.detach().cpu(),
        exposed_source_type=relation.source_type,
        exposed_relation=relation,
        diagnostics=diagnostics,
    )


def anchor_residual_decomposition(demand: torch.Tensor, anchor_message: torch.Tensor) -> AnchorResidualResult:
    if demand.shape != anchor_message.shape:
        raise ValueError("demand and anchor_message must have identical shape")
    residual = demand - anchor_message
    denom = torch.linalg.norm(demand).clamp_min(1e-12)
    return AnchorResidualResult(
        anchor_message=anchor_message,
        residual=residual,
        residual_energy_before=float(torch.linalg.norm(demand).item()),
        residual_energy_after=float((torch.linalg.norm(residual) / denom).item()),
    )
