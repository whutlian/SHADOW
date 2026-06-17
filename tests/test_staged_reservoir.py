from __future__ import annotations

import numpy as np
import torch

from shadow_hgc.reviewer_defense.staged_reservoir import build_staged_ranked_prefixes as reviewer_build_staged_ranked_prefixes
from shadow_hgc.sft.unified_objective import build_staged_ranked_prefixes


def test_staged_prefixes_are_nested_when_weights_change_by_budget() -> None:
    labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    rows = torch.arange(4, dtype=torch.long)
    features = np.zeros((4, 2), dtype=np.float32)
    teacher = np.asarray(
        [
            [0.50, 0.50],
            [0.99, 0.01],
            [0.80, 0.20],
            [0.70, 0.30],
        ],
        dtype=np.float32,
    )
    prefixes = build_staged_ranked_prefixes(
        labels=labels,
        train_rows=rows,
        feature_values=features,
        budgets=[1, 2],
        num_classes=1,
        seed=0,
        stage_selection_weights={
            1: {"soft": 0.0, "diversity": 0.0},
            2: {"soft": 1.0, "diversity": 0.0},
        },
        teacher_probs=teacher,
    )
    assert prefixes[1].tolist() == [0]
    assert prefixes[2].tolist() == [0, 1]
    assert prefixes[2][: prefixes[1].numel()].tolist() == prefixes[1].tolist()


def test_staged_prefixes_preserve_classwise_budget_growth() -> None:
    labels = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    rows = torch.arange(6, dtype=torch.long)
    features = np.zeros((6, 2), dtype=np.float32)
    prefixes = build_staged_ranked_prefixes(
        labels=labels,
        train_rows=rows,
        feature_values=features,
        budgets=[2, 4],
        num_classes=2,
        seed=4,
        stage_selection_weights={
            2: {"diversity": 0.0},
            4: {"diversity": 0.0},
        },
    )
    assert prefixes[2].numel() == 2
    assert prefixes[4].numel() == 4
    assert sorted(labels[prefixes[2]].tolist()) == [0, 1]
    assert prefixes[4][:2].tolist() == prefixes[2].tolist()


def test_reviewer_defense_export_uses_main_staged_builder() -> None:
    assert reviewer_build_staged_ranked_prefixes is build_staged_ranked_prefixes
