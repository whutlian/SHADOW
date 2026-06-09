from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from shadow_hgc.features.metapath_schema import schema_default_metapath_blocks
from shadow_hgc.features.path_label_affinity import compute_path_label_affinity, path_lad_diagnostics


@dataclass
class PathLADV2Result:
    blocks: dict[str, torch.Tensor]
    diagnostics: dict


def _train_count(train_target_mask: torch.Tensor, train_labels: torch.Tensor, num_classes: int) -> int:
    labels = train_labels.to(torch.long)
    valid = train_target_mask.to(torch.bool) & (labels >= 0) & (labels < int(num_classes))
    return int(valid.sum().item())


def _hub_clipped_edge_index(edge_index: torch.Tensor, num_source_nodes: int, quantile: float | None) -> tuple[torch.Tensor, int | None]:
    if quantile is None or quantile >= 1.0 or edge_index.numel() == 0:
        return edge_index, None
    src = edge_index[0].to(torch.long)
    degree = torch.bincount(src, minlength=int(num_source_nodes)).to(torch.float32)
    active = degree[degree > 0]
    if active.numel() == 0:
        return edge_index, None
    threshold = max(1, int(torch.quantile(active, float(quantile)).ceil().item()))
    keep = degree[src] <= float(threshold)
    clipped = edge_index[:, keep]
    if clipped.shape[1] == 0:
        return edge_index, threshold
    return clipped, threshold


def compute_path_lad_v2_blocks(
    graph,
    *,
    requested_paths: list[str] | None,
    train_target_mask: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    target_nodes: torch.Tensor | None = None,
    leave_one_out: bool = True,
    row_normalize: bool = True,
    hub_clip_quantile: float | None = 0.99,
    log1p_count: bool = True,
    block_gate: bool = True,
) -> PathLADV2Result:
    defaults = schema_default_metapath_blocks(
        dataset_name=getattr(graph, "dataset_name", ""),
        target_type=graph.target_type,
        relations=list(graph.relations),
        requested_blocks=requested_paths,
    )
    blocks: dict[str, torch.Tensor] = {}
    block_dims: dict[str, int] = {}
    thresholds: dict[str, int | None] = {}
    block_stats: dict[str, dict] = {}
    for name, relation in defaults.relation_by_block.items():
        edge_index = graph.edge_index[relation]
        clipped_edge_index, threshold = _hub_clipped_edge_index(
            edge_index,
            int(graph.num_nodes[relation.source_type]),
            hub_clip_quantile,
        )
        thresholds[name] = threshold
        proxy = SimpleNamespace(
            edge_index={**graph.edge_index, relation: clipped_edge_index},
            num_nodes=graph.num_nodes,
        )
        block = compute_path_label_affinity(
            proxy,
            target_type=graph.target_type,
            path=[relation],
            train_target_mask=train_target_mask,
            train_labels=train_labels,
            num_classes=num_classes,
            target_nodes=target_nodes,
            leave_one_out_for_train=leave_one_out,
            normalize="row_l1" if row_normalize else "none",
        )
        if log1p_count:
            block = torch.log1p(block)
        blocks[name] = block
        block_dims[name] = int(block.shape[1])
        diag = path_lad_diagnostics(block, leave_one_out_for_train=leave_one_out).to_json()
        diag["row_normalize"] = bool(row_normalize)
        diag["hub_clip_threshold"] = threshold
        block_stats[name] = diag
    diagnostics = {
        "path_lad_blocks": list(blocks),
        "path_lad_block_dims": block_dims,
        "path_lad_hub_clip_thresholds": thresholds,
        "path_lad_num_train_labels_used": _train_count(train_target_mask, train_labels, num_classes),
        "path_lad_uses_val_or_test_labels": False,
        "path_lad_uses_train_labels_only": True,
        "path_lad_row_normalize": bool(row_normalize),
        "path_lad_leave_one_out": bool(leave_one_out),
        "path_lad_hub_clip_quantile": None if hub_clip_quantile is None else float(hub_clip_quantile),
        "path_lad_log1p_count": bool(log1p_count),
        "path_lad_block_norm_stats_source": "train_full_target_rows",
        "path_lad_gate_values": {name: 1.0 for name in blocks} if block_gate else {},
        "path_lad_skipped_blocks": defaults.skipped_blocks,
        "path_lad_block_stats": block_stats,
    }
    return PathLADV2Result(blocks=blocks, diagnostics=diagnostics)

