from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.reddit.condensed_graph_builder import CondensedGraph, destination_row_normalize_edges, _unique_edges


@dataclass(frozen=True)
class EdgePredictorTrainingSet:
    pair_index: torch.Tensor
    labels: torch.Tensor
    features: torch.Tensor
    diagnostics: dict[str, Any]


def _positive_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    if edge_index.numel() == 0:
        return set()
    return {(int(src), int(dst)) for src, dst in edge_index.t().cpu().tolist()}


def deterministic_negative_pairs(
    *,
    num_nodes: int,
    positive_edge_index: torch.Tensor,
    labels: torch.Tensor | None = None,
    num_pairs: int,
    seed: int,
) -> torch.Tensor:
    positives = _positive_set(positive_edge_index)
    candidates = [(i, j) for i in range(int(num_nodes)) for j in range(int(num_nodes)) if i != j and (i, j) not in positives]
    if labels is not None:
        y = labels.detach().to(torch.long).cpu()
        candidates.sort(key=lambda pair: (int(y[pair[0]].item()) == int(y[pair[1]].item()), pair[0], pair[1]), reverse=True)
    generator = torch.Generator().manual_seed(int(seed))
    if not candidates:
        return torch.empty((2, 0), dtype=torch.long)
    perm = torch.randperm(len(candidates), generator=generator)[: min(int(num_pairs), len(candidates))]
    chosen = [candidates[int(idx.item())] for idx in perm]
    return torch.tensor(chosen, dtype=torch.long).t().contiguous()


def build_pair_features(
    node_features: torch.Tensor,
    pair_index: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    degree: torch.Tensor | None = None,
    cooccur_score: torch.Tensor | None = None,
    use_same_label: bool = True,
) -> torch.Tensor:
    z = node_features.detach().to(torch.float32).cpu()
    pair = pair_index.detach().to(torch.long).cpu()
    if pair.ndim != 2 or pair.shape[0] != 2:
        raise ValueError("pair_index must have shape [2, P]")
    src = pair[0]
    dst = pair[1]
    z_src = z[src]
    z_dst = z[dst]
    pieces = [z_src, z_dst, (z_src - z_dst).abs(), z_src * z_dst]
    if degree is not None:
        deg = degree.detach().to(torch.float32).cpu().view(-1, 1)
        pieces.extend([deg[src], deg[dst]])
    cosine = torch.nn.functional.cosine_similarity(z_src, z_dst).view(-1, 1)
    l2 = torch.norm(z_src - z_dst, dim=1, p=2).view(-1, 1)
    pieces.extend([cosine, l2])
    if use_same_label and labels is not None:
        y = labels.detach().to(torch.long).cpu()
        pieces.append((y[src] == y[dst]).to(torch.float32).view(-1, 1))
    if cooccur_score is not None:
        pieces.append(cooccur_score.detach().to(torch.float32).cpu().view(-1, 1))
    return torch.cat(pieces, dim=1)


def build_edge_candidate_pairs(
    node_features: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    max_candidates_per_node: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    z = torch.nn.functional.normalize(node_features.detach().to(torch.float32).cpu(), dim=1)
    n = int(z.shape[0])
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long)
    k = min(max(1, int(max_candidates_per_node)), max(1, n - 1))
    sim = z @ z.t()
    sim.fill_diagonal_(-float("inf"))
    _, idx = torch.topk(sim, k=k, dim=1)
    pairs: list[tuple[int, int]] = []
    for dst in range(n):
        for src in idx[dst].tolist():
            if src != dst:
                pairs.append((int(src), int(dst)))
    if labels is not None and n > 1:
        y = labels.detach().to(torch.long).cpu()
        generator = torch.Generator().manual_seed(int(seed))
        for dst in range(n):
            same = torch.nonzero(y == y[dst], as_tuple=False).view(-1)
            same = same[same != dst]
            if same.numel():
                src = int(same[torch.randint(0, same.numel(), (1,), generator=generator)].item())
                pairs.append((src, dst))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    unique, _ = _unique_edges(edge_index)
    return unique


def edge_predictor_topk_graph(
    candidate_pairs: torch.Tensor,
    scores: torch.Tensor,
    *,
    num_nodes: int,
    topk: int,
    add_self_loops: bool = True,
) -> CondensedGraph:
    pairs = candidate_pairs.detach().to(torch.long).cpu()
    score = scores.detach().to(torch.float32).cpu().clamp(0.0, 1.0)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("candidate_pairs must have shape [2, P]")
    selected: list[tuple[int, int]] = []
    weights: list[float] = []
    for dst in range(int(num_nodes)):
        mask = pairs[1] == dst
        if not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        local_scores = score[idx]
        keep = idx[torch.argsort(local_scores, descending=True)[: int(topk)]]
        for pos in keep.tolist():
            selected.append((int(pairs[0, pos].item()), int(pairs[1, pos].item())))
            weights.append(float(score[pos].item()))
    if add_self_loops:
        for node in range(int(num_nodes)):
            selected.append((node, node))
            weights.append(1.0)
    edge_index = torch.tensor(selected, dtype=torch.long).t().contiguous() if selected else torch.empty((2, 0), dtype=torch.long)
    raw = torch.tensor(weights, dtype=torch.float32) if weights else torch.empty(0, dtype=torch.float32)
    edge_index, raw = _unique_edges(edge_index, raw)
    weight = destination_row_normalize_edges(edge_index, raw, num_nodes=int(num_nodes))
    return CondensedGraph(
        edge_index=edge_index,
        edge_weight=weight,
        metadata={
            "edge_builder": "edge_predictor",
            "uses_edge_predictor": True,
            "edge_predictor_train_pairs": int(candidate_pairs.shape[1]),
            "edge_predictor_pos_rate": "",
            "edge_topk": int(topk),
            "edge_weight_normalization": "dst_row",
            "condensed_edges": int(edge_index.shape[1]),
            "uses_full_edge_index_on_gpu": False,
            "uses_e_by_d_materialization": False,
        },
    )


def build_edge_predictor_training_set(
    node_features: torch.Tensor,
    positive_edge_index: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    degree: torch.Tensor | None = None,
    negative_ratio: int = 1,
    seed: int = 42,
) -> EdgePredictorTrainingSet:
    pos = positive_edge_index.detach().to(torch.long).cpu()
    neg = deterministic_negative_pairs(
        num_nodes=int(node_features.shape[0]),
        positive_edge_index=pos,
        labels=labels,
        num_pairs=max(1, int(pos.shape[1]) * int(negative_ratio)),
        seed=int(seed),
    )
    pair_index = torch.cat([pos, neg], dim=1) if neg.numel() else pos
    y = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]).to(torch.float32)
    features = build_pair_features(node_features, pair_index, labels=labels, degree=degree, use_same_label=labels is not None)
    return EdgePredictorTrainingSet(
        pair_index=pair_index,
        labels=y,
        features=features,
        diagnostics={
            "edge_predictor_train_pairs": int(pair_index.shape[1]),
            "edge_predictor_pos_rate": float(y.mean().item()) if y.numel() else 0.0,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
        },
    )
