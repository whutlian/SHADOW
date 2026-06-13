from __future__ import annotations

from typing import Any

import torch

from shadow_hgc.sft.teacher_transport import TTCCondensedTable
from shadow_hgc.sft.ttcpp_selection_streaming import ratio_adaptive_v2_budget, select_ratio_adaptive_v2


def select_stt_streaming(
    *,
    features: torch.Tensor,
    teacher_probs: torch.Tensor,
    labels: torch.Tensor | None,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor | None = None,
    test_idx: torch.Tensor | None = None,
    num_rows: int,
    ratio: float,
    seed: int = 42,
    candidate_nodes: str = "all",
    reservoir_mode: str = "class_balanced",
    **kwargs: Any,
) -> TTCCondensedTable:
    table = select_ratio_adaptive_v2(
        features=features,
        teacher_probs=teacher_probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=int(num_rows),
        ratio=float(ratio),
        seed=int(seed),
        virtual_mixup_enabled=bool(kwargs.get("virtual_mixup_enabled", False)),
        virtual_mixup_count=int(kwargs.get("virtual_mixup_count", 0)),
    )
    budget = ratio_adaptive_v2_budget(num_rows=int(num_rows), ratio=float(ratio))
    diag = dict(table.diagnostics)
    diag.update(
        {
            "candidate_nodes_mode": str(candidate_nodes),
            "reservoir_mode": str(reservoir_mode),
            "uses_global_sort": False,
            "budget_policy": "ratio_adaptive_v2",
            "core_budget": budget.get("core", 0),
            "boundary_budget": budget.get("boundary", 0),
            "rare_budget": budget.get("rare", 0),
            "uncertainty_budget": budget.get("disagreement", 0),
            "total_condensed_nodes": int(num_rows),
        }
    )
    table.diagnostics.clear()
    table.diagnostics.update(diag)
    return table
