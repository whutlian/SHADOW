from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.features.label_affinity import (
    aggregate_non_target_label_affinity,
    compute_source_label_counts,
    compute_target_target_label_affinity,
    normalize_label_affinity_block,
)


@dataclass
class PathLabelAffinityDiagnostics:
    zero_row_ratio: float
    mean_l1_norm: float
    mean_entropy: float
    uses_train_labels_only: bool
    leave_one_out_for_train: bool

    def to_json(self) -> dict:
        return {
            "zero_row_ratio": float(self.zero_row_ratio),
            "mean_l1_norm": float(self.mean_l1_norm),
            "mean_entropy": float(self.mean_entropy),
            "uses_train_labels_only": bool(self.uses_train_labels_only),
            "leave_one_out_for_train": bool(self.leave_one_out_for_train),
        }


def _as_relation(value: DirectedRelation | tuple[str, str, str]) -> DirectedRelation:
    if isinstance(value, DirectedRelation):
        return value
    return DirectedRelation(*value)


def _graph_edge_index(graph) -> dict[DirectedRelation, torch.Tensor]:
    if isinstance(graph, dict):
        return graph["edge_index"]
    return graph.edge_index


def _graph_num_nodes(graph) -> dict[str, int]:
    if isinstance(graph, dict):
        return graph["num_nodes"]
    return graph.num_nodes


def _safe_train_labels(train_target_mask: torch.Tensor, train_labels: torch.Tensor) -> torch.Tensor:
    labels = train_labels.to(torch.long).clone()
    safe = torch.full_like(labels, -1)
    mask = train_target_mask.to(dtype=torch.bool, device=labels.device)
    safe[mask] = labels[mask]
    return safe


def _target_node_rows(num_nodes: int, target_nodes: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if target_nodes is None:
        return torch.arange(num_nodes, dtype=torch.long, device=device)
    return target_nodes.to(device=device, dtype=torch.long)


def _propagate_target_labels_along_relation(
    labels_block: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_dst_nodes: int,
) -> torch.Tensor:
    device = labels_block.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    out = torch.zeros(num_dst_nodes, labels_block.shape[1], dtype=labels_block.dtype, device=device)
    if edge_index.numel() == 0:
        return out
    alpha = destination_row_normalize(edge_index, num_dst_nodes).to(device=device, dtype=labels_block.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    out.index_add_(0, dst, labels_block[src] * alpha.unsqueeze(1))
    return out


def _target_target_path_affinity(
    *,
    graph,
    target_type: str,
    path: list[DirectedRelation],
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    target_nodes: torch.Tensor | None,
    leave_one_out_for_train: bool,
) -> torch.Tensor:
    edge_index_by_relation = _graph_edge_index(graph)
    num_target_nodes = int(_graph_num_nodes(graph)[target_type])
    safe_labels = _safe_train_labels(train_target_mask, train_labels)
    block = torch.zeros(num_target_nodes, num_classes, dtype=torch.float32, device=safe_labels.device)
    valid = (safe_labels >= 0) & (safe_labels < num_classes)
    if bool(valid.any()):
        block[valid, safe_labels[valid]] = 1.0
    for relation in path:
        if relation.source_type != target_type or relation.destination_type != target_type:
            raise ValueError("multi-step target-target Path-LAD requires target-target relations")
        block = _propagate_target_labels_along_relation(
            block,
            edge_index_by_relation[relation],
            num_dst_nodes=num_target_nodes,
        )
    rows = _target_node_rows(num_target_nodes, target_nodes, block.device)
    out = block[rows].clone()
    if leave_one_out_for_train:
        train_mask = train_target_mask.to(device=block.device, dtype=torch.bool)
        labels = safe_labels.to(device=block.device, dtype=torch.long)
        local_train = train_mask[rows] & (labels[rows] >= 0) & (labels[rows] < num_classes)
        if bool(local_train.any()):
            local_rows = torch.nonzero(local_train, as_tuple=False).flatten()
            out[local_rows, labels[rows[local_train]]] = 0.0
    return out


def _path_label_affinity_raw(
    graph,
    *,
    target_type: str,
    path: list[DirectedRelation],
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    target_nodes: torch.Tensor | None,
    leave_one_out_for_train: bool,
) -> torch.Tensor:
    if not path:
        raise ValueError("path must contain at least one relation")
    edge_index_by_relation = _graph_edge_index(graph)
    num_nodes = _graph_num_nodes(graph)
    relation = path[-1]
    if relation.destination_type != target_type:
        raise ValueError("Path-LAD path must end at the target type")
    safe_labels = _safe_train_labels(train_target_mask, train_labels)

    if relation.source_type == target_type:
        return _target_target_path_affinity(
            graph=graph,
            target_type=target_type,
            path=path,
            train_target_mask=train_target_mask,
            train_labels=safe_labels,
            num_classes=num_classes,
            target_nodes=target_nodes,
            leave_one_out_for_train=leave_one_out_for_train,
        )

    target_rows = _target_node_rows(int(num_nodes[target_type]), target_nodes, edge_index_by_relation[relation].device)
    source_affinity = compute_source_label_counts(
        edge_index_by_relation[relation],
        train_target_mask.to(device=edge_index_by_relation[relation].device),
        safe_labels.to(device=edge_index_by_relation[relation].device),
        num_source_nodes=int(num_nodes[relation.source_type]),
        num_classes=num_classes,
    )
    return aggregate_non_target_label_affinity(
        edge_index_by_relation[relation],
        source_affinity,
        target_nodes=target_rows,
        target_train_labels=safe_labels.to(device=edge_index_by_relation[relation].device),
        alpha=destination_row_normalize(edge_index_by_relation[relation], int(num_nodes[target_type])),
        leave_one_out_for_train=leave_one_out_for_train,
    )


def path_lad_diagnostics(block: torch.Tensor, *, leave_one_out_for_train: bool, eps: float = 1e-12) -> PathLabelAffinityDiagnostics:
    l1 = block.abs().sum(dim=1)
    prob = block / l1.unsqueeze(1).clamp_min(eps)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=1)
    return PathLabelAffinityDiagnostics(
        zero_row_ratio=(int((l1 <= eps).sum().item()) / int(l1.numel())) if l1.numel() else 0.0,
        mean_l1_norm=float(l1.mean().item()) if l1.numel() else 0.0,
        mean_entropy=float(entropy.mean().item()) if entropy.numel() else 0.0,
        uses_train_labels_only=True,
        leave_one_out_for_train=bool(leave_one_out_for_train),
    )


def compute_path_label_affinity(
    graph,
    target_type: str,
    path: list[DirectedRelation | tuple[str, str, str]],
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    target_nodes: torch.Tensor | None = None,
    leave_one_out_for_train: bool = True,
    normalize: str = "row_l1",
) -> torch.Tensor:
    """Return a train-label-only Path-LAD feature block for target rows."""

    relations = [_as_relation(item) for item in path]
    raw = _path_label_affinity_raw(
        graph,
        target_type=target_type,
        path=relations,
        train_target_mask=train_target_mask,
        train_labels=train_labels,
        num_classes=int(num_classes),
        target_nodes=target_nodes,
        leave_one_out_for_train=leave_one_out_for_train,
    )
    normalized, _ = normalize_label_affinity_block(raw, mode=normalize)
    return normalized
