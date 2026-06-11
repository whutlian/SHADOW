from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class CondensedGraph:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    metadata: dict[str, Any]


def _unique_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        return edge_index.to(torch.long), torch.empty(0, dtype=torch.float32)
    pairs = edge_index.t().to(torch.long).cpu()
    weights = torch.ones(pairs.shape[0], dtype=torch.float32) if edge_weight is None else edge_weight.detach().to(torch.float32).cpu()
    merged: dict[tuple[int, int], float] = {}
    for (src, dst), weight in zip(pairs.tolist(), weights.tolist()):
        key = (int(src), int(dst))
        merged[key] = max(float(weight), merged.get(key, 0.0))
    keys = sorted(merged)
    out_index = torch.tensor(keys, dtype=torch.long).t().contiguous()
    out_weight = torch.tensor([merged[key] for key in keys], dtype=torch.float32)
    return out_index, out_weight


def ensure_self_loops(edge_index: torch.Tensor, *, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(int(num_nodes), dtype=torch.long)
    loop_index = torch.stack([loops, loops], dim=0)
    if edge_index.numel() == 0:
        return loop_index
    merged, _ = _unique_edges(torch.cat([edge_index.to(torch.long).cpu(), loop_index], dim=1))
    return merged


def destination_row_normalize_edges(edge_index: torch.Tensor, raw_weight: torch.Tensor | None, *, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    weight = torch.ones(edge_index.shape[1], dtype=torch.float32) if raw_weight is None else raw_weight.to(torch.float32).cpu()
    return destination_row_normalize(edge_index.to(torch.long).cpu(), int(num_nodes), raw_edge_weight=weight)


def _normalize_graph(edge_index: torch.Tensor, raw_weight: torch.Tensor, *, num_nodes: int, metadata: dict[str, Any]) -> CondensedGraph:
    edge_index, raw_weight = _unique_edges(edge_index, raw_weight)
    edge_weight = destination_row_normalize_edges(edge_index, raw_weight.clamp_min(0.0), num_nodes=int(num_nodes))
    return CondensedGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        metadata={**metadata, "edge_weight_normalization": "dst_row", "condensed_edges": int(edge_index.shape[1])},
    )


def build_knn_graph(
    features: torch.Tensor,
    *,
    topk: int,
    metric: str = "cosine",
    symmetrize: str = "union",
    add_self_loops: bool = True,
    eps: float = 1e-6,
) -> CondensedGraph:
    x = features.detach().to(torch.float32).cpu()
    n = int(x.shape[0])
    if n == 0:
        return CondensedGraph(torch.empty((2, 0), dtype=torch.long), torch.empty(0), {"edge_builder": "knn"})
    k = max(0, min(int(topk), max(0, n - 1)))
    if metric == "cosine":
        normed = torch.nn.functional.normalize(x, p=2, dim=1, eps=eps)
        score = normed @ normed.t()
        score.fill_diagonal_(-float("inf"))
        raw_value = score.clamp_min(0.0) + eps
        largest = True
    elif metric == "l2":
        dist = torch.cdist(x, x, p=2)
        dist.fill_diagonal_(float("inf"))
        score = -dist
        raw_value = 1.0 / (1.0 + dist.clamp_min(0.0))
        largest = True
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")

    edge_pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    if k > 0:
        values, indices = torch.topk(score, k=k, dim=1, largest=largest)
        for dst in range(n):
            for src in indices[dst].tolist():
                if src == dst:
                    continue
                edge_pairs.append((int(src), int(dst)))
                weights.append(float(raw_value[dst, src].item()))
                if symmetrize == "union":
                    edge_pairs.append((int(dst), int(src)))
                    weights.append(float(raw_value[dst, src].item()))
                elif symmetrize == "mutual_knn":
                    pass
                elif symmetrize not in {"none", ""}:
                    raise ValueError("symmetrize must be union, mutual_knn, or none")
    if add_self_loops:
        for node in range(n):
            edge_pairs.append((node, node))
            weights.append(1.0)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    raw_weight = torch.tensor(weights, dtype=torch.float32) if weights else torch.empty(0, dtype=torch.float32)
    return _normalize_graph(
        edge_index,
        raw_weight,
        num_nodes=n,
        metadata={
            "edge_builder": "knn",
            "uses_knn_graph": True,
            "metric": metric,
            "edge_topk": int(topk),
            "edge_symmetry": symmetrize,
            "knn_signature_dim": int(x.shape[1]),
            "uses_full_edge_index_on_gpu": False,
            "uses_e_by_d_materialization": False,
        },
    )


def build_retained_edge_graph(
    *,
    selected_node_ids: torch.Tensor,
    edge_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    add_self_loops: bool = True,
) -> CondensedGraph:
    selected = selected_node_ids.detach().to(torch.long).cpu()
    mapping = {int(node): idx for idx, node in enumerate(selected.tolist())}
    pairs: list[tuple[int, int]] = []
    scans = 0
    for src_chunk, dst_chunk in edge_chunks:
        scans = 1
        src = src_chunk.detach().to(torch.long).cpu()
        dst = dst_chunk.detach().to(torch.long).cpu()
        for s, d in zip(src.tolist(), dst.tolist()):
            if int(s) in mapping and int(d) in mapping:
                pairs.append((mapping[int(s)], mapping[int(d)]))
    retained = len(pairs)
    if add_self_loops:
        pairs.extend((idx, idx) for idx in range(int(selected.numel())))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    raw_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return _normalize_graph(
        edge_index,
        raw_weight,
        num_nodes=int(selected.numel()),
        metadata={
            "edge_builder": "cooccur",
            "uses_cooccur_graph": True,
            "cooccur_sketch_type": "direct_retained_edge",
            "cooccur_sketch_size": 0,
            "retained_original_edges": int(retained),
            "estimated_cooccurrence_edges": 0,
            "full_edge_scans": int(scans),
            "loads_edge_index": False,
            "uses_full_edge_index_on_gpu": False,
            "uses_e_by_d_materialization": False,
        },
    )


def build_cooccurrence_sketch_graph(
    *,
    selected_node_ids: torch.Tensor,
    edge_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    topk: int,
    sketch_size: int = 1024,
    add_self_loops: bool = True,
) -> CondensedGraph:
    """Build a bounded selected-prototype co-occurrence graph by streaming edges.

    The sketches store hashed neighbor ids for selected nodes only. This keeps
    memory bounded by the number of selected prototypes and avoids materializing
    Reddit's full adjacency or moving a full edge index to GPU.
    """

    selected = selected_node_ids.detach().to(torch.long).cpu()
    mapping = {int(node): idx for idx, node in enumerate(selected.tolist())}
    n = int(selected.numel())
    sketches: list[set[int]] = [set() for _ in range(n)]
    direct_pairs: list[tuple[int, int]] = []
    scans = 0
    modulo = max(1, int(sketch_size))
    for src_chunk, dst_chunk in edge_chunks:
        scans = 1
        src = src_chunk.detach().to(torch.long).cpu()
        dst = dst_chunk.detach().to(torch.long).cpu()
        for s_value, d_value in zip(src.tolist(), dst.tolist()):
            s = int(s_value)
            d = int(d_value)
            if s in mapping:
                sketches[mapping[s]].add(hash(d) % modulo)
            if d in mapping:
                sketches[mapping[d]].add(hash(s) % modulo)
            if s in mapping and d in mapping:
                direct_pairs.append((mapping[s], mapping[d]))

    edge_pairs = list(direct_pairs)
    weights = [1.0 for _ in direct_pairs]
    k = max(0, int(topk))
    estimated = 0
    for dst in range(n):
        scores: list[tuple[float, int]] = []
        dst_sketch = sketches[dst]
        if not dst_sketch:
            continue
        for src in range(n):
            if src == dst:
                continue
            src_sketch = sketches[src]
            if not src_sketch:
                continue
            inter = len(dst_sketch.intersection(src_sketch))
            union = len(dst_sketch.union(src_sketch))
            if union == 0 or inter == 0:
                continue
            scores.append((float(inter) / float(union), src))
        scores.sort(key=lambda item: (-item[0], item[1]))
        for score, src in scores[:k]:
            edge_pairs.append((src, dst))
            weights.append(score)
            estimated += 1

    if add_self_loops:
        for node in range(n):
            edge_pairs.append((node, node))
            weights.append(1.0)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    raw_weight = torch.tensor(weights, dtype=torch.float32) if weights else torch.empty(0, dtype=torch.float32)
    return _normalize_graph(
        edge_index,
        raw_weight,
        num_nodes=n,
        metadata={
            "edge_builder": "cooccur",
            "uses_cooccur_graph": True,
            "cooccur_sketch_type": "hashed_neighbor_jaccard",
            "cooccur_sketch_size": int(sketch_size),
            "cooccur_topk": int(topk),
            "retained_original_edges": int(len(direct_pairs)),
            "estimated_cooccurrence_edges": int(estimated),
            "full_edge_scans": int(scans),
            "loads_edge_index": False,
            "uses_full_edge_index_on_gpu": False,
            "uses_e_by_d_materialization": False,
        },
    )
