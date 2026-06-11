from __future__ import annotations

import torch

from shadow_hgc.sft.teacher_transport import build_ttc_condensed_table, teacher_probability_diagnostics


def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.1],
            [0.0, 2.0],
            [0.1, 1.8],
            [1.0, 1.0],
            [1.1, 1.0],
            [0.9, 1.2],
            [2.2, 0.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.05, 0.95],
            [0.10, 0.90],
            [0.52, 0.48],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.99, 0.01],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 0])
    train_idx = torch.tensor([0, 2, 4])
    valid_idx = torch.tensor([1, 3])
    test_idx = torch.tensor([5, 6, 7])
    return features, probs, labels, train_idx, valid_idx, test_idx


def test_t31_teacher_probability_diagnostics_reports_simplex_stats() -> None:
    _, probs, *_ = _toy()
    diag = teacher_probability_diagnostics(probs)
    assert diag["teacher_entropy_mean"] > 0.0
    assert diag["teacher_margin_mean"] > 0.0
    assert diag["predicted_classes"] == 2


def test_t31_ttc_selection_uses_all_nodes_but_hard_anchors_train_only() -> None:
    features, probs, labels, train_idx, valid_idx, test_idx = _toy()
    result = build_ttc_condensed_table(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=6,
        mode="ttc_coverage_plus_boundary",
        seed=7,
    )
    assert result.z_syn.shape == (6, 2)
    assert torch.allclose(result.y_syn_soft.sum(dim=1), torch.ones(6))
    selected_train = set(result.source_node_ids[result.hard_anchor_mask].tolist())
    assert selected_train <= set(train_idx.tolist())
    assert result.diagnostics["candidate_nodes"] == "all"
    assert sum(result.diagnostics["selected_bucket_counts"].values()) == 6


def test_t31_ttc_selection_ignores_valid_and_test_labels() -> None:
    features, probs, labels, train_idx, valid_idx, test_idx = _toy()
    first = build_ttc_condensed_table(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=5,
        mode="ttc_confidence_balanced",
        seed=3,
    )
    changed = labels.clone()
    changed[valid_idx] = 1 - changed[valid_idx]
    changed[test_idx] = 1 - changed[test_idx]
    second = build_ttc_condensed_table(
        features=features,
        teacher_probs=probs,
        labels=changed,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=5,
        mode="ttc_confidence_balanced",
        seed=3,
    )
    assert first.source_node_ids.tolist() == second.source_node_ids.tolist()
    assert first.bucket_types == second.bucket_types


def test_t31_ttc_mixup_preserves_soft_label_simplex() -> None:
    features, probs, labels, train_idx, valid_idx, test_idx = _toy()
    result = build_ttc_condensed_table(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=6,
        mode="ttc_coverage_plus_boundary_plus_mixup",
        seed=11,
        mixup_alpha=0.4,
    )
    assert result.diagnostics["mixup_row_count"] > 0
    assert torch.allclose(result.y_syn_soft.sum(dim=1), torch.ones(6), atol=1e-6)
