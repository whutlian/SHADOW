from __future__ import annotations

import torch

from shadow_hgc.sft.correct_smooth import correct_and_smooth
from shadow_hgc.sft.timeaware_arxiv_v2 import (
    build_timeaware_arxiv_features,
    temporal_labelreuse_decay_v2,
    train_year_class_prior_features,
)


def _toy_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [3.0, 0.1, -0.2],
            [0.2, 2.0, 0.0],
            [0.0, 0.1, 2.5],
            [0.5, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )


def test_t28_correct_and_smooth_ignores_valid_and_test_labels():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    train_idx = torch.tensor([0, 1])
    valid_idx = torch.tensor([2])
    test_idx = torch.tensor([3])
    labels_a = torch.tensor([0, 1, 2, 0])
    labels_b = torch.tensor([0, 1, 0, 2])
    out_a = correct_and_smooth(
        _toy_logits(),
        labels_a,
        train_idx,
        valid_idx,
        test_idx,
        edge_index,
        num_classes=3,
        correction_alpha=0.5,
        smoothing_alpha=0.5,
        num_correction_steps=2,
        num_smoothing_steps=2,
    )
    out_b = correct_and_smooth(
        _toy_logits(),
        labels_b,
        train_idx,
        valid_idx,
        test_idx,
        edge_index,
        num_classes=3,
        correction_alpha=0.5,
        smoothing_alpha=0.5,
        num_correction_steps=2,
        num_smoothing_steps=2,
    )
    assert torch.allclose(out_a.logits_or_probs, out_b.logits_or_probs)
    assert out_a.diagnostics["uses_valid_labels_as_input"] is False
    assert out_a.diagnostics["uses_test_labels_as_input"] is False
    assert out_a.diagnostics["creates_dense_adjacency"] is False
    assert out_a.diagnostics["normalization"] == "dst_row"


def test_t28_correct_and_smooth_changes_when_train_label_changes():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    train_idx = torch.tensor([0, 1])
    labels_a = torch.tensor([0, 1, 2, 0])
    labels_b = torch.tensor([1, 1, 2, 0])
    out_a = correct_and_smooth(
        _toy_logits(),
        labels_a,
        train_idx,
        torch.tensor([2]),
        torch.tensor([3]),
        edge_index,
        num_classes=3,
        correction_alpha=0.5,
        smoothing_alpha=0.0,
        num_correction_steps=1,
        num_smoothing_steps=0,
    )
    out_b = correct_and_smooth(
        _toy_logits(),
        labels_b,
        train_idx,
        torch.tensor([2]),
        torch.tensor([3]),
        edge_index,
        num_classes=3,
        correction_alpha=0.5,
        smoothing_alpha=0.0,
        num_correction_steps=1,
        num_smoothing_steps=0,
    )
    assert not torch.allclose(out_a.logits_or_probs, out_b.logits_or_probs)


def test_t28_temporal_features_use_train_labels_only():
    years = torch.tensor([2016, 2017, 2018, 2019])
    labels_a = torch.tensor([0, 1, 2, 2])
    labels_b = torch.tensor([0, 1, 0, 1])
    train_idx = torch.tensor([0, 1])
    valid_idx = torch.tensor([2])
    test_idx = torch.tensor([3])
    prior_a = train_year_class_prior_features(years, labels_a, train_idx, num_classes=3)
    prior_b = train_year_class_prior_features(years, labels_b, train_idx, num_classes=3)
    assert torch.allclose(prior_a, prior_b)
    features = build_timeaware_arxiv_features(years, labels_a, train_idx, valid_idx, test_idx, num_classes=3)
    assert features.features.shape[0] == 4
    assert features.diagnostics["uses_valid_labels_as_input"] is False
    assert features.diagnostics["uses_test_labels_as_input"] is False


def test_t28_temporal_decay_gamma_changes_label_reuse_weights():
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    years = torch.tensor([2016, 2017, 2019, 2019])
    labels = torch.tensor([0, 1, 2, 2])
    train_idx = torch.tensor([0, 1])
    low = temporal_labelreuse_decay_v2(edge_index, years, labels, train_idx, num_classes=3, gamma=0.01)
    high = temporal_labelreuse_decay_v2(edge_index, years, labels, train_idx, num_classes=3, gamma=0.20)
    assert not torch.allclose(low.raw_weights, high.raw_weights)
    assert low.diagnostics["uses_train_labels_only"] is True
