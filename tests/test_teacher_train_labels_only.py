from __future__ import annotations

import torch

from shadow_hgc.teacher.train_teacher import build_train_only_teacher_targets


def test_teacher_targets_ignore_validation_and_test_labels():
    train_idx = torch.tensor([0, 2])
    labels_a = torch.tensor([1, 0, 2, 1])
    labels_b = torch.tensor([1, 2, 2, 0])

    first = build_train_only_teacher_targets(labels_a, train_idx)
    second = build_train_only_teacher_targets(labels_b, train_idx)

    assert torch.equal(first, second)
    assert first.tolist() == [1, -1, 2, -1]
