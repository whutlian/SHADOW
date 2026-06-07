from __future__ import annotations

from pathlib import Path

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation


TARGET_TYPES = {
    "acm": "paper",
    "dblp": "author",
    "imdb": "movie",
}

RELATION_NAME_MAP = {
    ("paper", "cite", "paper"): "cite_ref",
    ("paper", "ref", "paper"): "cited_by",
    ("author", "to", "paper"): "writes",
    ("subject", "to", "paper"): "subject_of",
    ("term", "to", "paper"): "term_in",
    ("paper", "to", "author"): "written_by",
    ("director", "to", "movie"): "directs",
    ("actor", "to", "movie"): "acts_in",
    ("keyword", "to", "movie"): "keyword_in",
}


def _processed_path(dataset: str, root: str | Path) -> Path:
    return Path(root) / dataset / "processed" / "data.pt"


def _node_count(store: dict, node_type: str, data: dict) -> int:
    if "x" in store:
        return int(store["x"].shape[0])
    if "num_nodes" in store:
        return int(store["num_nodes"])
    max_id = -1
    for key, edge_store in data.items():
        if not isinstance(key, tuple):
            continue
        src_type, _, dst_type = key
        edge_index = edge_store["edge_index"]
        if src_type == node_type and edge_index.numel() > 0:
            max_id = max(max_id, int(edge_index[0].max().item()))
        if dst_type == node_type and edge_index.numel() > 0:
            max_id = max(max_id, int(edge_index[1].max().item()))
    return max_id + 1


def _relation_name(src_type: str, raw_name: str, dst_type: str) -> str:
    return RELATION_NAME_MAP.get((src_type, raw_name, dst_type), f"{src_type}_{raw_name}_{dst_type}")


def load_processed_small_dataset(dataset: str, *, root: str | Path = "dataset") -> HeteroGraphData:
    dataset = dataset.lower()
    if dataset not in TARGET_TYPES:
        raise ValueError(f"unknown small dataset: {dataset}")
    path = _processed_path(dataset, root)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    data = obj[0] if isinstance(obj, tuple) else obj
    target_type = TARGET_TYPES[dataset]

    node_features: dict[str, torch.Tensor] = {}
    num_nodes: dict[str, int] = {}
    for node_type, store in data.items():
        if not isinstance(node_type, str) or node_type == "_global_store":
            continue
        num_nodes[node_type] = _node_count(store, node_type, data)
        if "x" in store:
            node_features[node_type] = store["x"].to(torch.float32)

    target_store = data[target_type]
    labels = target_store["y"]
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    labels = labels.to(torch.long)
    train_idx = torch.nonzero(target_store["train_mask"], as_tuple=False).flatten().to(torch.long)
    test_idx = torch.nonzero(target_store["test_mask"], as_tuple=False).flatten().to(torch.long)
    val_idx = torch.empty(0, dtype=torch.long)

    relations: list[DirectedRelation] = []
    edge_index: dict[DirectedRelation, torch.Tensor] = {}
    for key, edge_store in data.items():
        if not isinstance(key, tuple):
            continue
        src_type, raw_name, dst_type = key
        if dst_type != target_type:
            continue
        relation = DirectedRelation(src_type, _relation_name(src_type, raw_name, dst_type), dst_type)
        relations.append(relation)
        edge_index[relation] = edge_store["edge_index"].to(torch.long)

    if not relations:
        raise ValueError(f"{dataset} has no incoming relations to target type {target_type}")

    return HeteroGraphData(
        dataset_name=dataset,
        target_type=target_type,
        node_features=node_features,
        edge_index=edge_index,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        relations=relations,
        num_nodes=num_nodes,
    )
