from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


SAFE_ASSIGNMENT_MODES = {
    "qoc_class_conditional_online_kmeans",
    "qoc_sft_ctc_assignment",
    "qoc_sft_bonsai_assignment",
    "qoc_hybrid_assignment",
}

PLTC_ASSIGNMENT_MODES = {
    "qoc_pltc_confidence_balanced",
    "qoc_pltc_uncertainty_balanced",
    "qoc_pltc_class_mass_balanced",
}


@dataclass(frozen=True)
class CodebookAssignmentResult:
    assignments: torch.Tensor
    codebook_features: torch.Tensor
    codebook_train_label_mass: torch.Tensor
    codebook_soft_label_mass: torch.Tensor | None
    codebook_node_mass: torch.Tensor
    codebook_degree_stats: dict[str, Any]
    codebook_origin_stats: dict[str, Any]
    diagnostics: dict[str, Any]


def safe_qoc_budget_split(num_codewords: int) -> dict[str, int]:
    total = int(num_codewords)
    labeled = int(round(total * 0.70))
    unlabeled = int(round(total * 0.20))
    rare = max(0, total - labeled - unlabeled)
    return {"labeled_train_codewords": labeled, "unlabeled_structural_codewords": unlabeled, "residual_rare_codewords": rare}


def _initial_centers(features: torch.Tensor, labels: torch.Tensor | None, train_idx: torch.Tensor | None, num_codewords: int, num_classes: int, seed: int) -> torch.Tensor:
    x = features.to(torch.float32).cpu()
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("features must contain at least one node")
    generator = torch.Generator().manual_seed(int(seed))
    centers: list[torch.Tensor] = []
    if labels is not None and train_idx is not None and train_idx.numel() > 0:
        y = labels.to(torch.long).cpu()
        train = train_idx.to(torch.long).cpu()
        for cls in range(int(num_classes)):
            cls_rows = train[y[train] == cls]
            if cls_rows.numel() > 0:
                centers.append(x[cls_rows].mean(dim=0))
    remaining = int(num_codewords) - len(centers)
    if remaining > 0:
        perm = torch.randperm(n, generator=generator)[: min(remaining, n)]
        centers.extend(x[perm])
    while len(centers) < int(num_codewords):
        centers.append(x.mean(dim=0))
    return torch.stack(centers[: int(num_codewords)]).to(torch.float32)


def _assign_nearest(features: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    x = features.to(torch.float32).cpu()
    c = centers.to(torch.float32).cpu()
    out = torch.empty(x.shape[0], dtype=torch.long)
    for start in range(0, x.shape[0], int(chunk_size)):
        block = x[start : start + int(chunk_size)]
        out[start : start + block.shape[0]] = torch.cdist(block, c).argmin(dim=1)
    return out


def _means_by_assignment(features: torch.Tensor, assignments: torch.Tensor, num_codewords: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = features.to(torch.float32).cpu()
    a = assignments.to(torch.long).cpu()
    sums = torch.zeros(int(num_codewords), x.shape[1], dtype=torch.float32)
    counts = torch.bincount(a, minlength=int(num_codewords)).to(torch.float32)
    sums.index_add_(0, a, x)
    nonzero = counts > 0
    sums[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)
    if bool((~nonzero).any()):
        sums[~nonzero] = x.mean(dim=0)
    return sums, counts


def _train_label_mass(assignments: torch.Tensor, labels: torch.Tensor | None, train_idx: torch.Tensor | None, num_codewords: int, num_classes: int) -> torch.Tensor:
    mass = torch.zeros(int(num_codewords), int(num_classes), dtype=torch.float32)
    if labels is None or train_idx is None or train_idx.numel() == 0:
        return mass
    train = train_idx.to(torch.long).cpu()
    code = assignments[train].to(torch.long)
    y = labels.to(torch.long).cpu()[train]
    valid = (y >= 0) & (y < int(num_classes))
    if bool(valid.any()):
        flat = code[valid] * int(num_classes) + y[valid]
        mass += torch.bincount(flat, minlength=int(num_codewords) * int(num_classes)).view(int(num_codewords), int(num_classes)).to(torch.float32)
    return mass


def build_codebook_assignment(
    *,
    features: torch.Tensor,
    labels: torch.Tensor | None = None,
    train_idx: torch.Tensor | None = None,
    num_codewords: int,
    num_classes: int,
    mode: str,
    seed: int = 42,
    soft_labels: torch.Tensor | None = None,
    degree: torch.Tensor | None = None,
    chunk_size: int = 4096,
    refine_steps: int = 2,
) -> CodebookAssignmentResult:
    started = time.perf_counter()
    if mode not in SAFE_ASSIGNMENT_MODES | PLTC_ASSIGNMENT_MODES:
        raise ValueError(f"unsupported T30 QOC assignment mode: {mode}")
    centers = _initial_centers(features, labels, train_idx, int(num_codewords), int(num_classes), int(seed))
    assignments = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(max(1, int(refine_steps))):
        assignments = _assign_nearest(features, centers, int(chunk_size))
        centers, counts = _means_by_assignment(features, assignments, int(num_codewords))
    codebook_features, counts = _means_by_assignment(features, assignments, int(num_codewords))
    train_mass = _train_label_mass(assignments, labels, train_idx, int(num_codewords), int(num_classes))
    soft_mass = None
    if soft_labels is not None:
        soft = soft_labels.to(torch.float32).cpu()
        soft_mass = torch.zeros(int(num_codewords), soft.shape[1], dtype=torch.float32)
        soft_mass.index_add_(0, assignments, soft)
        nonzero = counts > 0
        soft_mass[nonzero] = soft_mass[nonzero] / counts[nonzero].unsqueeze(1)
    degree_stats: dict[str, Any] = {}
    if degree is not None:
        d = degree.to(torch.float32).cpu()
        degree_stats = {
            "global_degree_mean": float(d.mean().item()) if d.numel() else 0.0,
            "global_degree_max": float(d.max().item()) if d.numel() else 0.0,
        }
    labeled = int((train_mass.sum(dim=1) > 0).sum().item())
    diagnostics = {
        "assignment_coverage": 1.0,
        "num_assigned_nodes": int(features.shape[0]),
        "num_unassigned_nodes": int((assignments < 0).sum().item()),
        "codewords_with_train_label_mass": labeled,
        "codewords_without_train_label_mass": int(num_codewords) - labeled,
        "min_codeword_mass": float(counts.min().item()) if counts.numel() else 0.0,
        "median_codeword_mass": float(counts.median().item()) if counts.numel() else 0.0,
        "max_codeword_mass": float(counts.max().item()) if counts.numel() else 0.0,
        "assignment_time": float(time.perf_counter() - started),
        "assignment_peak_ram": "",
        "assignment_mode": mode,
    }
    return CodebookAssignmentResult(
        assignments=assignments.to(torch.int32),
        codebook_features=codebook_features,
        codebook_train_label_mass=train_mass,
        codebook_soft_label_mass=soft_mass,
        codebook_node_mass=counts,
        codebook_degree_stats=degree_stats,
        codebook_origin_stats={"mode": mode, **safe_qoc_budget_split(int(num_codewords))},
        diagnostics=diagnostics,
    )
