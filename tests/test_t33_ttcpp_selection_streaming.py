from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_selection_streaming import (
    ratio_adaptive_v2_budget,
    select_ratio_adaptive_v2,
)


def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.eye(8, dtype=torch.float32)
    probs = torch.tensor(
        [
            [0.98, 0.02],
            [0.95, 0.05],
            [0.05, 0.95],
            [0.10, 0.90],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.70, 0.30],
            [0.30, 0.70],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    train = torch.tensor([0, 2, 4])
    valid = torch.tensor([1, 3])
    test = torch.tensor([5, 6, 7])
    return features, probs, labels, train, valid, test


def test_t33_ratio_adaptive_v2_budget_changes_with_scale() -> None:
    small = ratio_adaptive_v2_budget(num_rows=100, ratio=0.001)
    mid = ratio_adaptive_v2_budget(num_rows=100, ratio=0.0025)
    large = ratio_adaptive_v2_budget(num_rows=100, ratio=0.005)
    assert small["core"] == 70
    assert small["train_hard_anchor"] == 7
    assert mid["core"] == 55
    assert large["core"] == 40
    assert large["boundary"] > small["boundary"]


def test_t33_selection_is_invariant_to_valid_and_test_labels() -> None:
    features, probs, labels, train, valid, test = _toy()
    first = select_ratio_adaptive_v2(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train,
        valid_idx=valid,
        test_idx=test,
        num_rows=5,
        ratio=0.001,
        seed=7,
    )
    changed = labels.clone()
    changed[valid] = 1 - changed[valid]
    changed[test] = 1 - changed[test]
    second = select_ratio_adaptive_v2(
        features=features,
        teacher_probs=probs,
        labels=changed,
        train_idx=train,
        valid_idx=valid,
        test_idx=test,
        num_rows=5,
        ratio=0.001,
        seed=7,
    )
    assert first.source_node_ids.tolist() == second.source_node_ids.tolist()
    assert torch.allclose(first.y_syn_soft, second.y_syn_soft)
    assert first.diagnostics["candidate_nodes_mode"] == "all"


def test_t33_virtual_mixup_does_not_count_as_condensed_rows() -> None:
    features, probs, labels, train, valid, test = _toy()
    table = select_ratio_adaptive_v2(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train,
        valid_idx=valid,
        test_idx=test,
        num_rows=4,
        ratio=0.005,
        seed=7,
        virtual_mixup_enabled=True,
        virtual_mixup_count=12,
    )
    assert table.z_syn.shape[0] == 4
    assert table.diagnostics["total_condensed_nodes"] == 4
    assert table.diagnostics["virtual_mixup_count"] == 12
