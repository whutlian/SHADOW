from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

import torch

from shadow_hgc.features.metapath_schema import schema_default_metapath_blocks


def _label_distribution(labels: torch.Tensor, idx: torch.Tensor) -> dict[str, int]:
    if idx.numel() == 0:
        return {}
    selected = labels[idx].to(torch.long)
    counts = Counter(int(value.item()) for value in selected if int(value.item()) >= 0)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def _tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def split_hash(graph) -> str:
    payload = {
        "train": graph.train_idx.detach().cpu().tolist(),
        "valid": graph.val_idx.detach().cpu().tolist(),
        "test": graph.test_idx.detach().cpu().tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def feature_hash(graph) -> str:
    digest = hashlib.sha256()
    for node_type in sorted(graph.node_features):
        digest.update(node_type.encode("utf-8"))
        digest.update(graph.node_features[node_type].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def label_hash(graph) -> str:
    return _tensor_hash(graph.labels)


def schema_hash(graph) -> str:
    payload = {
        "target_type": graph.target_type,
        "num_nodes": {key: int(value) for key, value in sorted(graph.num_nodes.items())},
        "relations": [
            [relation.source_type, relation.relation_name, relation.destination_type, int(graph.edge_index[relation].shape[1])]
            for relation in sorted(graph.relations)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _edge_type_rows(graph) -> list[dict[str, Any]]:
    return [
        {
            "source_type": relation.source_type,
            "relation_name": relation.relation_name,
            "destination_type": relation.destination_type,
            "num_edges": int(graph.edge_index[relation].shape[1]),
        }
        for relation in graph.relations
    ]


def _relation_pairs(graph) -> set[tuple[str, str]]:
    return {(relation.source_type, relation.destination_type) for relation in graph.relations}


def _path_available(graph, block: str) -> bool:
    pairs = _relation_pairs(graph)
    target = graph.target_type
    if block == "APA":
        return ("author", "paper") in pairs and ("paper", "author") in pairs or ("paper", "author") in pairs
    if block in {"APVPA", "APCPA"}:
        return ("author", "paper") in pairs and (("paper", "venue") in pairs or ("paper", "conference") in pairs) and (("venue", "paper") in pairs or ("conference", "paper") in pairs) and ("paper", "author") in pairs
    if block == "APTPA":
        return ("author", "paper") in pairs and ("paper", "term") in pairs and ("term", "paper") in pairs and ("paper", "author") in pairs
    if block == "PAP":
        return ("paper", "author") in pairs and ("author", "paper") in pairs or ("author", "paper") in pairs
    if block == "PSP":
        return ("paper", "subject") in pairs and ("subject", "paper") in pairs or ("subject", "paper") in pairs
    if block == "PTP":
        return ("paper", "term") in pairs and ("term", "paper") in pairs or ("term", "paper") in pairs
    if block == "MAM":
        return ("movie", "actor") in pairs and ("actor", "movie") in pairs or ("actor", "movie") in pairs
    if block == "MDM":
        return ("movie", "director") in pairs and ("director", "movie") in pairs or ("director", "movie") in pairs
    if block == "MKM":
        return ("movie", "keyword") in pairs and ("keyword", "movie") in pairs or ("keyword", "movie") in pairs
    return False


def _expected_metapaths(dataset: str) -> list[str]:
    if dataset == "dblp":
        return ["APA", "APVPA", "APTPA", "APCPA"]
    if dataset == "acm":
        return ["PAP", "PSP", "PTP"]
    if dataset == "imdb":
        return ["MAM", "MDM", "MKM"]
    return []


def audit_schema_alignment(graph, *, loader_name: str, source: str) -> dict[str, Any]:
    dataset = graph.dataset_name.lower()
    expected = _expected_metapaths(dataset)
    available = [name for name in expected if _path_available(graph, name)]
    missing = [name for name in expected if name not in available]
    if not expected:
        status = "not_applicable"
    elif missing:
        status = "partial"
    else:
        status = "aligned"
    if loader_name == "current_processed" and dataset == "dblp" and missing:
        missing_reason = "current loader keeps incoming-to-target relations, so richer DBLP meta-path support is incomplete"
    elif missing:
        missing_reason = "required schema edge pairs are absent in the loaded graph"
    else:
        missing_reason = ""
    return {
        "dataset": dataset,
        "source": source,
        "loader_name": loader_name,
        "target_type": graph.target_type,
        "label_node_type": graph.target_type,
        "node_types": sorted(graph.num_nodes),
        "edge_types": _edge_type_rows(graph),
        "num_nodes_by_type": {key: int(value) for key, value in sorted(graph.num_nodes.items())},
        "num_edges_by_type": {str(relation): int(graph.edge_index[relation].shape[1]) for relation in graph.relations},
        "metapath_available": available,
        "metapath_missing": missing,
        "missing_reason": missing_reason,
        "split_hash": split_hash(graph),
        "feature_hash": feature_hash(graph),
        "label_hash": label_hash(graph),
        "schema_hash": schema_hash(graph),
        "freehgc_or_hgb_alignment_status": status,
        "notes": "full_schema audit only; default condensation path remains incoming-to-target",
    }


def audit_dblp_schema(graph, requested_metapath_blocks: list[str] | None = None) -> dict[str, Any]:
    defaults = schema_default_metapath_blocks(
        dataset_name="dblp",
        target_type=graph.target_type,
        relations=list(graph.relations),
        requested_blocks=requested_metapath_blocks,
    )
    edge_types = _edge_type_rows(graph)
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
