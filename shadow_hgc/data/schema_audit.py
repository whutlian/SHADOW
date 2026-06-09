from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from shadow_hgc.features.metapath_schema import schema_default_metapath_blocks


def _label_distribution(labels: torch.Tensor, idx: torch.Tensor) -> dict[str, int]:
    if idx.numel() == 0:
        return {}
    selected = labels[idx].to(torch.long)
    counts = Counter(int(value.item()) for value in selected if int(value.item()) >= 0)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def audit_dblp_schema(graph, requested_metapath_blocks: list[str] | None = None) -> dict[str, Any]:
    defaults = schema_default_metapath_blocks(
        dataset_name="dblp",
        target_type=graph.target_type,
        relations=list(graph.relations),
        requested_blocks=requested_metapath_blocks,
    )
    edge_types = [
        {
            "source_type": relation.source_type,
            "relation_name": relation.relation_name,
            "destination_type": relation.destination_type,
            "num_edges": int(graph.edge_index[relation].shape[1]),
        }
        for relation in graph.relations
    ]
    available_sources = sorted({relation.source_type for relation in graph.relations if relation.destination_type == graph.target_type})
    target_is_author = graph.target_type == "author"
    apa_available = "APA" in defaults.available_blocks
    no_paper_centered_author_mismatch = all(name.startswith("A") and name.endswith("A") for name in defaults.available_blocks)
    return {
        "dataset": "dblp",
        "target_type": graph.target_type,
        "label_node_type": graph.target_type,
        "node_types": sorted(graph.num_nodes),
        "node_counts": {node_type: int(count) for node_type, count in sorted(graph.num_nodes.items())},
        "edge_types": edge_types,
        "available_source_types_for_target_metapaths": available_sources,
        "requested_metapath_blocks": defaults.requested_blocks,
        "computed_metapath_blocks": defaults.available_blocks,
        "skipped_metapath_blocks": defaults.skipped_blocks,
        "apa_available": apa_available,
        "train_label_distribution": _label_distribution(graph.labels, graph.train_idx),
        "valid_label_distribution": _label_distribution(graph.labels, graph.val_idx),
        "test_label_distribution": _label_distribution(graph.labels, graph.test_idx),
        "hard_requirements_passed": bool(target_is_author and apa_available and no_paper_centered_author_mismatch),
        "notes": "APVPA/APTPA require non-target paper-venue/paper-term edges; current small loader keeps incoming-to-target relations only.",
    }

