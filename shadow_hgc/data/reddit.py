from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shadow_hgc.data.schemas import DirectedRelation


@dataclass(frozen=True)
class RedditGraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_idx: torch.Tensor
    valid_idx: torch.Tensor
    test_idx: torch.Tensor
    diagnostics: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1 if self.y.numel() else 0

    @property
    def target_type(self) -> str:
        return "reddit_node"

    @property
    def relation(self) -> DirectedRelation:
        return DirectedRelation("reddit_node", "links", "reddit_node")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _mask_to_idx(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).view(-1).to(torch.long)
    return mask.to(torch.long).view(-1)


def load_reddit_dataset(root: str | Path = "dataset/Reddit", *, self_loop: bool = False, load_tensors: bool = True) -> RedditGraphData:
    if self_loop:
        raise ValueError("T24 Reddit loader keeps --reddit-self-loop false by default; self loops are not implemented here")
    base = Path(root)
    processed = base / "processed" / "data.pt"
    if not processed.exists():
        raise FileNotFoundError(f"Reddit processed cache not found: {processed}")
    if not load_tensors:
        raise ValueError("load_tensors=False is not supported for the processed PyG cache")
    obj = torch.load(processed, map_location="cpu", weights_only=False)
    data = obj[0] if isinstance(obj, tuple) else obj
    if not isinstance(data, dict):
        data = data.to_dict()
    x = data["x"].to(torch.float32).cpu()
    edge_index = data["edge_index"].to(torch.long).cpu()
    y = data["y"].to(torch.long).view(-1).cpu()
    train_idx = _mask_to_idx(data["train_mask"].cpu())
    valid_idx = _mask_to_idx(data.get("val_mask", data.get("valid_mask")).cpu())
    test_idx = _mask_to_idx(data["test_mask"].cpu())
    diagnostics = {
        "source": "processed_pyg_cache",
        "path": str(processed),
        "hash": _hash_file(processed),
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "feature_dim": int(x.shape[1]),
        "train_nodes": int(train_idx.numel()),
        "valid_nodes": int(valid_idx.numel()),
        "test_nodes": int(test_idx.numel()),
        "self_loop": False,
    }
    return RedditGraphData(x=x, edge_index=edge_index, y=y, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, diagnostics=diagnostics)


def reddit_graph_spec(graph: RedditGraphData) -> dict[str, Any]:
    return {
        "target_type": graph.target_type,
        "relations": {graph.relation: graph.edge_index},
        "num_nodes": {graph.target_type: graph.num_nodes},
    }
