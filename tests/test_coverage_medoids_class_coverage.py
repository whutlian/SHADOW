from __future__ import annotations

import torch

from shadow_hgc.prototype.coverage_medoids import select_coverage_medoids


def test_coverage_medoids_select_at_least_one_per_available_class():
    result = select_coverage_medoids(
        signatures=torch.randn(8, 4),
        labels=torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]),
        train_idx=torch.arange(8),
        total_budget=3,
        seed=42,
    )

    assert set(result.labels.tolist()) == {0, 1, 2}
    assert result.weights.shape == result.indices.shape
