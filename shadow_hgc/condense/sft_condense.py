from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from shadow_hgc.prototype.sft_herding import select_sft_herding, sqrt_class_budget
from shadow_hgc.shadows.assign import topb_nonnegative_assignment


@dataclass(frozen=True)
class SFTCondenseResult:
    selected_rows: torch.Tensor
    condensed_blocks: dict[str, torch.Tensor]
    condensed_labels: torch.Tensor
    method: str
    ratio: float
    diagnostics: dict[str, object]


def _class_rows(labels: torch.Tensor, train_rows: torch.Tensor, cls: int) -> torch.Tensor:
    return train_rows[labels[train_rows].to(torch.long) == int(cls)]


def _centroids(blocks: Mapping[str, torch.Tensor], labels: torch.Tensor, train_rows: torch.Tensor, total_budget: int) -> SFTCondenseResult:
    budget = sqrt_class_budget(labels, train_rows, total_budget)
    out_blocks: dict[str, list[torch.Tensor]] = {name: [] for name in blocks}
    out_labels: list[int] = []
    for cls, k in budget.items():
        rows = _class_rows(labels, train_rows, cls)
        if rows.numel() == 0:
            continue
        chunks = torch.chunk(rows, min(int(k), int(rows.numel())))
        for chunk in chunks:
            for name, block in blocks.items():
                out_blocks[name].append(block[chunk].to(torch.float32).mean(dim=0))
            out_labels.append(int(cls))
    condensed = {name: torch.stack(values, dim=0) if values else torch.empty(0, blocks[name].shape[1]) for name, values in out_blocks.items()}
    return SFTCondenseResult(
        selected_rows=torch.empty(0, dtype=torch.long),
        condensed_blocks=condensed,
        condensed_labels=torch.tensor(out_labels, dtype=torch.long),
        method="centroid",
        ratio=float(total_budget) / max(1, int(train_rows.numel())),
        diagnostics={"real_target_rows": False, "classwise_budget": True, "budget_by_class": budget},
    )


def nonnegative_b2_weights(demand: torch.Tensor, shadows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assignment = topb_nonnegative_assignment(demand.to(torch.float32), shadows.to(torch.float32), b=2)
    weights = assignment.topk_weight.to(torch.float32)
    indices = assignment.topk_index.to(torch.long)
    return indices, weights


def condense_sft_blocks(
    *,
    blocks: Mapping[str, torch.Tensor],
    signatures: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    ratio: float,
    method: Literal["centroid", "medoid", "herding"] = "herding",
    b: int = 1,
    train_losses: torch.Tensor | None = None,
    seed: int = 42,
) -> SFTCondenseResult:
    total_budget = max(int(labels.max().item()) + 1 if labels.numel() else 1, int(round(float(train_rows.numel()) * float(ratio))))
    if method == "centroid":
        result = _centroids(blocks, labels, train_rows, total_budget)
    else:
        selection = select_sft_herding(
            signatures=signatures,
            labels=labels,
            train_rows=train_rows,
            total_budget=total_budget,
            mode="medoid" if method == "medoid" else "herding",
            train_losses=train_losses,
            seed=seed,
        )
        condensed = {name: block[selection.selected_rows].to(torch.float32).clone() for name, block in blocks.items()}
        result = SFTCondenseResult(
            selected_rows=selection.selected_rows,
            condensed_blocks=condensed,
            condensed_labels=selection.labels,
            method=method,
            ratio=float(total_budget) / max(1, int(train_rows.numel())),
            diagnostics={**selection.diagnostics, "real_target_rows": True, "budget_by_class": selection.budget_by_class},
        )
    diag = {**result.diagnostics, "b": int(b), "b2_weights_nonnegative": True}
    if int(b) == 2 and result.condensed_blocks:
        first = next(iter(result.condensed_blocks.values()))
        indices, weights = nonnegative_b2_weights(first, first)
        diag.update({"b2_index_shape": [int(v) for v in indices.shape], "b2_weight_min": float(weights.min().item()) if weights.numel() else 0.0})
    return SFTCondenseResult(
        selected_rows=result.selected_rows,
        condensed_blocks=result.condensed_blocks,
        condensed_labels=result.condensed_labels,
        method=f"{method}_b{int(b)}",
        ratio=result.ratio,
        diagnostics=diag,
    )
