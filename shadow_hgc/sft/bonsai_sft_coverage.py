from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BonsaiCoverageResult:
    selected_idx: torch.Tensor
    coverage_counts: torch.Tensor
    diagnostics: dict[str, Any]


def _projection_codes(features: torch.Tensor, seed: int, dim: int = 16) -> torch.Tensor:
    x = features.detach().float().cpu()
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    proj = torch.randn(x.shape[1], min(dim, max(1, x.shape[1])), generator=g)
    signs = (x @ proj) > 0
    powers = (2 ** torch.arange(signs.shape[1])).long()
    return (signs.long() * powers).sum(dim=1)


def select_bonsai_coverage(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    num_rows: int,
    mode: str = "hard_train_label_coverage",
    teacher_probs: torch.Tensor | None = None,
    seed: int = 42,
) -> BonsaiCoverageResult:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    x = features.detach().float().cpu()
    train = train_idx.detach().cpu().long()
    uses_teacher = mode in {"soft_ttc_coverage", "coverage_plus_boundary"} or teacher_probs is not None and mode != "hard_train_label_coverage"
    if mode == "hard_train_label_coverage":
        candidates = train
        promotion_track = "safe_main"
        uses_teacher = False
    else:
        candidates = torch.arange(x.shape[0], dtype=torch.long)
        promotion_track = "sota_chase"
    codes = _projection_codes(x, seed=seed)
    all_counts = torch.bincount(codes, minlength=int(codes.max().item()) + 1).float()
    candidate_counts = all_counts[codes[candidates]].clone()
    if mode == "coverage_plus_boundary" and teacher_probs is not None:
        probs = teacher_probs.detach().float().cpu()
        top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
        candidate_counts = candidate_counts + (1.0 - margin[candidates].clamp(0.0, 1.0))
    order = torch.argsort(candidate_counts, descending=True, stable=True)
    selected = candidates[order[: min(num_rows, candidates.numel())]]
    if selected.numel() < num_rows:
        repeats = selected.repeat((num_rows + max(1, selected.numel()) - 1) // max(1, selected.numel())) if selected.numel() else candidates[:1].repeat(num_rows)
        selected = repeats[:num_rows]
    coverage = all_counts[codes[selected]].float()
    diagnostics = {
        "promotion_track": promotion_track,
        "uses_teacher_logits": bool(uses_teacher),
        "uses_exact_pairwise": False,
        "coverage_backend": "lsh",
        "candidate_nodes": "train" if mode == "hard_train_label_coverage" else "all",
        "candidate_count": int(candidates.numel()),
        "selected_count": int(selected.numel()),
        "selected_label_hist": torch.bincount(labels.detach().cpu().long()[selected], minlength=int(labels.max().item()) + 1).tolist()
        if selected.numel()
        else [],
    }
    return BonsaiCoverageResult(selected_idx=selected.long(), coverage_counts=coverage, diagnostics=diagnostics)
