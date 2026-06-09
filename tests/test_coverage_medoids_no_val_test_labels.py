from __future__ import annotations

import torch

from shadow_hgc.prototype.coverage_medoids import select_coverage_medoids


def test_coverage_medoids_ignore_labels_outside_train_idx():
    signatures = torch.randn(6, 3)
    train_idx = torch.tensor([0, 1, 2, 3])
    labels_a = torch.tensor([0, 0, 1, 1, 0, 1])
    labels_b = torch.tensor([0, 0, 1, 1, 1, 0])

    first = select_coverage_medoids(signatures, labels_a, train_idx=train_idx, total_budget=2, seed=7)
    second = select_coverage_medoids(signatures, labels_b, train_idx=train_idx, total_budget=2, seed=7)

    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.labels, second.labels)
