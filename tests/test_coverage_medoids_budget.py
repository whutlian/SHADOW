from __future__ import annotations

import torch

from shadow_hgc.prototype.coverage_medoids import select_coverage_medoids


def test_coverage_medoids_respect_class_budgets():
    result = select_coverage_medoids(
        signatures=torch.eye(6),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        train_idx=torch.arange(6),
        class_budget={0: 2, 1: 1},
        coverage_scores=torch.arange(6, dtype=torch.float32),
        boundary_scores=torch.zeros(6),
        seed=0,
    )

    assert result.indices.numel() == 3
    selected_labels = result.labels.tolist()
    assert selected_labels.count(0) == 2
    assert selected_labels.count(1) == 1
    assert result.diagnostics["prototype_mode"] == "coverage_medoid"
    assert result.diagnostics["medoid_real_node_ratio"] == 1.0
