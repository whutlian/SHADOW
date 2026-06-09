from __future__ import annotations

from dataclasses import dataclass

import torch

from shadow_hgc.features.metapath_schema import schema_default_metapath_blocks
from shadow_hgc.features.scap import (
    apply_hub_clipping,
    non_target_source_scap,
    prior_center_scap,
    source_class_affinity,
    target_target_scap_dense,
    transform_scap_block,
)


@dataclass(frozen=True)
class SCAPBlockResult:
    blocks: dict[str, torch.Tensor]
    diagnostics: dict[str, dict]


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return int(valid.max().item()) + 1 if valid.numel() else 0


def _train_mask(num_nodes: int, train_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    mask[train_idx.to(torch.long)] = True
    return mask


def build_scap_blocks_for_graph(
    graph,
    *,
    target_rows: torch.Tensor | None = None,
    prior_centering: bool = True,
    log1p: bool = True,
    l2_normalize: bool = False,
    hub_cap: int = 1024,
    include_path_scap: bool = True,
) -> SCAPBlockResult:
    target_type = graph.target_type
    labels = graph.labels.to(torch.long)
    num_classes = _num_classes(labels)
    rows = torch.arange(graph.num_nodes[target_type], dtype=torch.long) if target_rows is None else target_rows.to(torch.long)
    train_mask = _train_mask(graph.num_nodes[target_type], graph.train_idx)
    blocks: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict] = {}
    for relation in graph.relations:
        if relation.destination_type != target_type:
            continue
        edge_index = graph.edge_index[relation].to(torch.long)
        clipped, hub_meta = apply_hub_clipping(edge_index, hub_cap=hub_cap, policy="clip")
        name = f"scap:{relation.relation_name}"
        if relation.source_type == target_type:
            block = target_target_scap_dense(
                edge_index=clipped,
                labels=labels,
                train_mask=train_mask,
                num_nodes=graph.num_nodes[target_type],
                num_classes=num_classes,
                target_rows=rows,
            )
            diag = {
                "relation_or_path": str(relation),
                "source_type": relation.source_type,
                "dense_or_sparse": "dense",
                "num_target_rows": int(rows.numel()),
                "num_classes": int(num_classes),
                "uses_train_labels_only": True,
                "uses_validation_labels": False,
                "uses_test_labels": False,
                "full_edge_scans": 2,
                **hub_meta,
            }
        else:
            source_counts = source_class_affinity(
                source_to_target_edges=clipped,
                labels=labels,
                train_mask=train_mask,
                num_source_nodes=graph.num_nodes[relation.source_type],
                num_classes=num_classes,
                active_source_nodes=torch.unique(clipped[0]) if clipped.numel() else torch.empty(0, dtype=torch.long),
            )
            block, source_diag = non_target_source_scap(
                edge_index_source_to_target=clipped,
                source_affinity=source_counts,
                num_target_nodes=graph.num_nodes[target_type],
                target_rows=rows,
            )
            diag = {
                "relation_or_path": str(relation),
                "source_type": relation.source_type,
                "dense_or_sparse": "dense",
                "num_target_rows": int(rows.numel()),
                "num_classes": int(num_classes),
                "uses_train_labels_only": True,
                "uses_validation_labels": False,
                "uses_test_labels": False,
                "full_edge_scans": 2,
                **hub_meta,
                **source_diag,
            }
        if prior_centering:
            block, prior_meta = prior_center_scap(block, train_labels=labels[graph.train_idx], num_classes=num_classes)
            diag.update(prior_meta)
        else:
            diag["prior_centering"] = False
        block = transform_scap_block(block, log1p=log1p, l2_normalize=l2_normalize)
        diag["cache_bytes"] = int(block.numel() * 2)
        blocks[name] = block.to(torch.float16 if num_classes <= 200 else torch.float32)
        diagnostics[name] = diag
    if include_path_scap:
        defaults = schema_default_metapath_blocks(
            dataset_name=graph.dataset_name,
            target_type=target_type,
            relations=list(graph.relations),
        )
        for block_name, relation in defaults.relation_by_block.items():
            edge_index = graph.edge_index[relation].to(torch.long)
            clipped, hub_meta = apply_hub_clipping(edge_index, hub_cap=hub_cap, policy="clip")
            source_counts = source_class_affinity(
                source_to_target_edges=clipped,
                labels=labels,
                train_mask=train_mask,
                num_source_nodes=graph.num_nodes[relation.source_type],
                num_classes=num_classes,
                active_source_nodes=torch.unique(clipped[0]) if clipped.numel() else torch.empty(0, dtype=torch.long),
            )
            block, source_diag = non_target_source_scap(
                edge_index_source_to_target=clipped,
                source_affinity=source_counts,
                num_target_nodes=graph.num_nodes[target_type],
                target_rows=rows,
            )
            diag = {
                "relation_or_path": block_name,
                "source_relation": str(relation),
                "block_family": "path_scap",
                "path_length": 2,
                "dense_or_sparse": "dense",
                "num_target_rows": int(rows.numel()),
                "num_classes": int(num_classes),
                "uses_train_labels_only": True,
                "uses_validation_labels": False,
                "uses_test_labels": False,
                "full_edge_scans": 2,
                **hub_meta,
                **source_diag,
            }
            if prior_centering:
                block, prior_meta = prior_center_scap(block, train_labels=labels[graph.train_idx], num_classes=num_classes)
                diag.update(prior_meta)
            else:
                diag["prior_centering"] = False
            block = transform_scap_block(block, log1p=log1p, l2_normalize=l2_normalize)
            diag["cache_bytes"] = int(block.numel() * 2)
            name = f"path_scap:{block_name}"
            blocks[name] = block.to(torch.float16 if num_classes <= 200 else torch.float32)
            diagnostics[name] = diag
        for skipped in defaults.skipped_blocks:
            diagnostics[f"path_scap_skipped:{skipped}"] = {
                "relation_or_path": skipped,
                "block_family": "path_scap",
                "status": "skipped_schema_or_streaming_guard",
                "reason": "required source-target schema pair is unavailable or longer path is not streaming-safe in this desktop run",
                "uses_train_labels_only": True,
                "uses_validation_labels": False,
                "uses_test_labels": False,
                "full_edge_scans": 0,
            }
    return SCAPBlockResult(blocks=blocks, diagnostics=diagnostics)
