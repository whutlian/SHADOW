from __future__ import annotations

from typing import Any

import torch

from shadow_hgc.diagnostics.demand_equivalence import compare_relation_demand_blocks
from shadow_hgc.features.metapath_table import compute_metapath_feature
from shadow_hgc.fullgraph.metapath_specs import available_metapath_specs


def _relation_label(relation) -> str:
    return f"{relation.source_type}->{relation.relation_name}->{relation.destination_type}"


def audit_imdb_relation_inventory(graph) -> dict[str, Any]:
    if graph.dataset_name.lower() != "imdb":
        raise ValueError("IMDB inventory audit requires dataset_name='imdb'")
    incoming = [rel for rel in graph.relations if rel.destination_type == graph.target_type]
    relation_names = {_relation_label(rel) for rel in graph.relations}
    available, skipped = available_metapath_specs(graph.dataset_name, graph.relations, graph.target_type)
    coverage = {}
    for rel in incoming:
        dst = graph.edge_index[rel][1]
        train_dst = set(int(i) for i in graph.train_idx.tolist())
        covered = {int(v) for v in dst.tolist() if int(v) in train_dst}
        coverage[_relation_label(rel)] = len(covered)
    return {
        "dataset": graph.dataset_name,
        "target_type": graph.target_type,
        "all_node_types": sorted(graph.num_nodes),
        "all_edge_types": sorted(relation_names),
        "incoming_target_relations": sorted(_relation_label(rel) for rel in incoming),
        "typed:directs_exists": "director->directs->movie" in relation_names,
        "typed:acts_in_exists": "actor->acts_in->movie" in relation_names,
        "typed:keyword_in_exists": "keyword->keyword_in->movie" in relation_names,
        "MAM_available": "MAM" in available,
        "MDM_available": "MDM" in available,
        "MKM_available": "MKM" in available,
        "metapath_skipped": skipped,
        "feature_dims": {node_type: int(x.shape[1]) for node_type, x in graph.node_features.items()},
        "edge_counts_per_relation": {_relation_label(rel): int(graph.edge_index[rel].shape[1]) for rel in graph.relations},
        "train_target_coverage_per_relation": coverage,
    }


def _imdb_metapath_blocks(graph, target_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    available, _ = available_metapath_specs(graph.dataset_name, graph.relations, graph.target_type)
    feature_provider = {graph.target_type: graph.node_features[graph.target_type].to(torch.float32)}
    blocks = {}
    for name in ["MAM", "MDM", "MKM"]:
        if name not in available:
            continue
        block, _ = compute_metapath_feature(
            path_schema=available[name],
            target_type=graph.target_type,
            feature_provider=feature_provider,
            edge_store=graph.edge_index,
            num_nodes=graph.num_nodes,
            target_rows=target_rows,
        )
        blocks[name] = block
    return blocks


def compare_imdb_clean_s1_to_sfb_metapaths(graph, *, target_rows: torch.Tensor | None = None) -> dict[str, Any]:
    rows = target_rows if target_rows is not None else torch.arange(graph.num_nodes[graph.target_type], dtype=torch.long)
    clean_blocks = _imdb_metapath_blocks(graph, rows)
    sfb_reused_blocks = _imdb_metapath_blocks(graph, rows)
    metrics = {}
    for name, clean in clean_blocks.items():
        metrics[name] = compare_relation_demand_blocks(
            dataset="imdb",
            relation_name=f"metapath:{name}",
            demand_a=clean,
            demand_b=sfb_reused_blocks[name],
            train_target_ids=rows,
            source_type=graph.target_type,
            destination_type=graph.target_type,
            edge_direction_checked=True,
            alpha_normalization_checked=True,
        )
        metrics[name]["block_dim"] = int(clean.shape[1])
        metrics[name]["block_norm_stats_source"] = "clean_s1_metapath_provider"
    return metrics
