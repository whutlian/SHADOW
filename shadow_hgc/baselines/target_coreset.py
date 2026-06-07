from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.eval.logging import attach_run_metadata, write_json_summary
from shadow_hgc.features.projection import fit_standardizer, fixed_random_projection, standardize
from shadow_hgc.prototype.budgets import class_wise_budget


def _target_features(graph: HeteroGraphData, feature_dim: int, seed: int, projection_type: str) -> torch.Tensor:
    x = graph.node_features[graph.target_type].to(torch.float32)
    if projection_type == "raw":
        projected = x
    elif projection_type == "random":
        projected = fixed_random_projection(x, out_dim=feature_dim, seed=seed + 991)
    else:
        raise ValueError(f"unknown projection_type: {projection_type}")
    return standardize(projected, fit_standardizer(projected, rows=graph.train_idx))


def _select_random(x: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor, budgets: dict[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for cls, budget in budgets.items():
        candidates = train_idx[labels[train_idx] == cls]
        perm = torch.randperm(candidates.numel(), generator=generator)[: min(budget, candidates.numel())]
        selected.append(candidates[perm])
    return torch.cat(selected).to(torch.long)


def _select_herding(x: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor, budgets: dict[int, int]) -> torch.Tensor:
    selected = []
    for cls, budget in budgets.items():
        candidates = train_idx[labels[train_idx] == cls]
        center = x[candidates].mean(dim=0, keepdim=True)
        dist = torch.cdist(x[candidates], center).flatten()
        selected.append(candidates[torch.argsort(dist)[: min(budget, candidates.numel())]])
    return torch.cat(selected).to(torch.long)


def _select_kcenter(x: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor, budgets: dict[int, int], seed: int) -> torch.Tensor:
    selected = []
    generator = torch.Generator().manual_seed(seed)
    for cls, budget in budgets.items():
        candidates = train_idx[labels[train_idx] == cls]
        local = x[candidates]
        k = min(budget, candidates.numel())
        first = int(torch.randint(candidates.numel(), (1,), generator=generator).item())
        centers = [first]
        min_dist = torch.cdist(local, local[first : first + 1]).flatten()
        while len(centers) < k:
            nxt = int(torch.argmax(min_dist).item())
            centers.append(nxt)
            min_dist = torch.minimum(min_dist, torch.cdist(local, local[nxt : nxt + 1]).flatten())
        selected.append(candidates[torch.tensor(centers, dtype=torch.long)])
    return torch.cat(selected).to(torch.long)


def _train_linear(
    x: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    epochs: int,
    seed: int,
) -> tuple[float | None, float]:
    torch.manual_seed(seed)
    num_classes = int(labels[train_idx].max().item()) + 1
    model = torch.nn.Linear(x.shape[1], num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x[train_idx]), labels[train_idx])
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - start
    if test_idx.numel() == 0:
        return None, train_time
    pred = model(x).argmax(dim=1)
    return float((pred[test_idx] == labels[test_idx]).to(torch.float32).mean().item()), train_time


def run_target_coreset_baselines(
    graph: HeteroGraphData,
    *,
    seed: int,
    epochs: int,
    M_tau: int,
    feature_dim: int,
    projection_type: str = "random",
    log_dir: str | Path | None = None,
) -> list[dict]:
    x = _target_features(graph, feature_dim, seed, projection_type)
    budgets = class_wise_budget(graph.labels, graph.train_idx, M_tau)
    selectors = {
        "Random-HG": lambda: _select_random(x, graph.labels, graph.train_idx, budgets, seed),
        "Herding-HG": lambda: _select_herding(x, graph.labels, graph.train_idx, budgets),
        "K-Center-HG": lambda: _select_kcenter(x, graph.labels, graph.train_idx, budgets, seed),
    }
    rows = []
    for method, selector in selectors.items():
        selected = selector()
        accuracy, train_time = _train_linear(
            x,
            graph.labels,
            selected,
            graph.test_idx,
            epochs=epochs,
            seed=seed,
        )
        row = {
            "method": method,
            "dataset": graph.dataset_name,
            "target_type": graph.target_type,
            "mode": "target_feature_coreset_baseline",
            "requested_M_tau": M_tau,
            "effective_M_tau": int(selected.numel()),
            "feature_dim": feature_dim,
            "projection_type": projection_type,
            "seed": seed,
            "epochs": epochs,
            "accuracy": "" if accuracy is None else f"{accuracy:.6f}",
            "training_time": f"{train_time:.6f}",
            "condensation_time": 0.0,
            "inference_time": 0.0,
            "status": "completed",
        }
        config_for_hash = {
            "method": method,
            "dataset": graph.dataset_name,
            "seed": seed,
            "epochs": epochs,
            "M_tau": M_tau,
            "feature_dim": feature_dim,
            "projection_type": projection_type,
        }
        row = attach_run_metadata(row, config=config_for_hash)
        if log_dir is not None:
            output_path = Path(log_dir) / f"{graph.dataset_name}_{method.replace('-', '_')}_M{M_tau}_seed{seed}.json"
            write_json_summary(output_path, row, config=config_for_hash)
        rows.append(
            {
                "dataset": graph.dataset_name,
                "method": method,
                "mode": "target_feature_coreset_baseline",
                "M_tau": M_tau,
                "seed": seed,
                "accuracy": "" if accuracy is None else f"{accuracy:.6f}",
                "training_time": f"{train_time:.6f}",
                "status": "completed",
            }
        )
    return rows
