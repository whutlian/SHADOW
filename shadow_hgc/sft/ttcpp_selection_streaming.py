from __future__ import annotations

import json
import math
from typing import Any

import torch

from shadow_hgc.sft.teacher_transport import TTCCondensedTable
from shadow_hgc.sft.ttcpp_selector import select_ttc_rows_ratio_adaptive


def _allocate(num_rows: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: int(num_rows) * float(value) for key, value in weights.items()}
    alloc = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = int(num_rows) - sum(alloc.values())
    order = sorted(raw, key=lambda key: (raw[key] - alloc[key], raw[key]), reverse=True)
    for key in order[:remaining]:
        alloc[key] += 1
    return alloc


def ratio_adaptive_v2_budget(*, num_rows: int, ratio: float) -> dict[str, int]:
    ratio = float(ratio)
    if ratio <= 0.001:
        weights = {
            "core": 0.70,
            "boundary": 0.10,
            "rare": 0.08,
            "train_hard_anchor": 0.07,
            "prior_repair": 0.05,
            "disagreement": 0.0,
        }
    elif ratio <= 0.0025:
        weights = {
            "core": 0.55,
            "boundary": 0.18,
            "disagreement": 0.10,
            "rare": 0.08,
            "prior_repair": 0.05,
            "train_hard_anchor": 0.04,
        }
    else:
        weights = {
            "core": 0.40,
            "boundary": 0.25,
            "disagreement": 0.15,
            "rare": 0.10,
            "prior_repair": 0.05,
            "train_hard_anchor": 0.05,
        }
    return _allocate(int(num_rows), weights)


def _stats(values: torch.Tensor) -> tuple[float, float, float]:
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    return float(values.min().item()), float(values.float().median().item()), float(values.max().item())


def _enrich_diagnostics(table: TTCCondensedTable, *, ratio: float, budget: dict[str, int], virtual_mixup_enabled: bool, virtual_mixup_count: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row_type in table.bucket_types:
        key = str(row_type)
        if key == "rare_structure":
            key = "rare"
        counts[key] = counts.get(key, 0) + 1
    total = max(1, int(table.z_syn.shape[0]))
    pred = table.y_syn_soft.argmax(dim=1)
    selected_counts = torch.bincount(pred, minlength=table.y_syn_soft.shape[1])
    soft_mass = table.y_syn_soft.sum(dim=0)
    class_min, class_median, class_max = _stats(selected_counts[selected_counts > 0])
    mass_min, mass_median, mass_max = _stats(soft_mass[soft_mass > 0])
    diag = dict(table.diagnostics)
    diag.update(
        {
            "candidate_nodes_mode": "all",
            "budget_policy": "ratio_adaptive_v2",
            "budget_allocation_json": json.dumps(budget, sort_keys=True),
            "total_condensed_nodes": total,
            "core_frac_actual": counts.get("core", 0) / total,
            "boundary_frac_actual": counts.get("boundary", 0) / total,
            "disagreement_frac_actual": counts.get("disagreement", 0) / total,
            "rare_frac_actual": counts.get("rare", 0) / total,
            "anchor_frac_actual": counts.get("train_hard_anchor", int(table.hard_anchor_mask.sum().item())) / total,
            "prior_repair_frac_actual": counts.get("prior_repair", 0) / total,
            "selected_rows_per_class_min": class_min,
            "selected_rows_per_class_median": class_median,
            "selected_rows_per_class_max": class_max,
            "soft_class_mass_per_class_min": mass_min,
            "soft_class_mass_per_class_median": mass_median,
            "soft_class_mass_per_class_max": mass_max,
            "selected_soft_prior_kl_to_teacher_prior": diag.get("selected_soft_prior_kl", ""),
            "selected_hard_anchor_count": int(table.hard_anchor_mask.sum().item()),
            "signature_bucket_coverage": diag.get("degree_bucket_coverage", ""),
            "virtual_mixup_enabled": bool(virtual_mixup_enabled),
            "virtual_mixup_count": int(virtual_mixup_count),
            "requested_full_node_ratio": float(ratio),
        }
    )
    return diag


def select_ratio_adaptive_v2(
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
    disagreement: torch.Tensor | None = None,
    virtual_mixup_enabled: bool = False,
    virtual_mixup_count: int = 0,
) -> TTCCondensedTable:
    budget = ratio_adaptive_v2_budget(num_rows=int(num_rows), ratio=float(ratio))
    policy = "ratio_adaptive_core70" if ratio <= 0.001 else ("ratio_adaptive_core55" if ratio <= 0.0025 else "ratio_adaptive_core40")
    table = select_ttc_rows_ratio_adaptive(
        features=features,
        teacher_probs=teacher_probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=int(num_rows),
        ratio=float(ratio),
        policy=policy,
        seed=int(seed),
        disagreement=disagreement,
    )
    diag = _enrich_diagnostics(
        table,
        ratio=float(ratio),
        budget=budget,
        virtual_mixup_enabled=bool(virtual_mixup_enabled),
        virtual_mixup_count=int(virtual_mixup_count),
    )
    table.diagnostics.clear()
    table.diagnostics.update(diag)
    return table
