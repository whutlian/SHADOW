from __future__ import annotations

import torch

from shadow_hgc.sft.bonsai_sft_coverage import select_bonsai_coverage


def test_t31_bonsai_lsh_coverage_is_deterministic_and_nonnegative() -> None:
    features = torch.arange(24, dtype=torch.float32).view(8, 3)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    train_idx = torch.tensor([0, 1, 2, 3])
    first = select_bonsai_coverage(features=features, labels=labels, train_idx=train_idx, num_rows=3, mode="hard_train_label_coverage", seed=5)
    second = select_bonsai_coverage(features=features, labels=labels, train_idx=train_idx, num_rows=3, mode="hard_train_label_coverage", seed=5)
    assert first.selected_idx.tolist() == second.selected_idx.tolist()
    assert all(value >= 0 for value in first.coverage_counts.tolist())
    assert first.diagnostics["uses_exact_pairwise"] is False


def test_t31_bonsai_safe_mode_uses_only_train_labels_no_teacher_logits() -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    train_idx = torch.tensor([0, 1, 2, 3])
    result = select_bonsai_coverage(features=features, labels=labels, train_idx=train_idx, num_rows=4, mode="hard_train_label_coverage", seed=1)
    assert set(result.selected_idx.tolist()) <= set(train_idx.tolist())
    assert result.diagnostics["promotion_track"] == "safe_main"
    assert result.diagnostics["uses_teacher_logits"] is False


def test_t31_bonsai_sota_mode_logs_teacher_logits() -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    train_idx = torch.tensor([0, 1, 2, 3])
    teacher_probs = torch.softmax(torch.randn(8, 2), dim=1)
    result = select_bonsai_coverage(
        features=features,
        labels=labels,
        train_idx=train_idx,
        num_rows=4,
        mode="soft_ttc_coverage",
        teacher_probs=teacher_probs,
        seed=1,
    )
    assert result.diagnostics["promotion_track"] == "sota_chase"
    assert result.diagnostics["uses_teacher_logits"] is True
