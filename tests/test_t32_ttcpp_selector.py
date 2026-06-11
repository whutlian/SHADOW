from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_selector import (
    build_ratio_adaptive_budget,
    compute_selected_soft_prior,
    repair_soft_class_prior,
    select_ttc_rows_ratio_adaptive,
    soft_prior_kl,
)


def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [2.0, 0.0],
            [1.9, 0.1],
            [0.0, 2.0],
            [0.1, 1.9],
            [1.0, 1.0],
            [1.1, 1.0],
            [0.9, 1.1],
            [2.2, 0.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor(
        [
            [0.98, 0.02],
            [0.92, 0.08],
            [0.04, 0.96],
            [0.10, 0.90],
            [0.52, 0.48],
            [0.58, 0.42],
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


def test_t32_ratio_adaptive_budget_is_core_heavy_at_tiny_ratios() -> None:
    tiny = build_ratio_adaptive_budget(num_rows=100, ratio=0.001, policy="ratio_adaptive_core70")
    large = build_ratio_adaptive_budget(num_rows=100, ratio=0.005, policy="ratio_adaptive_core40")
    assert tiny["core"] >= 70
    assert tiny.get("mixup", 0) == 0
    assert large["boundary"] + large["disagreement"] > tiny["boundary"] + tiny["disagreement"]


def test_t32_selection_ignores_valid_and_test_labels_and_counts_budget() -> None:
    features, probs, labels, train_idx, valid_idx, test_idx = _toy()
    first = select_ttc_rows_ratio_adaptive(
        features=features,
        teacher_probs=probs,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=5,
        ratio=0.001,
        policy="ratio_adaptive_core70",
        seed=4,
    )
    changed = labels.clone()
    changed[valid_idx] = 1 - changed[valid_idx]
    changed[test_idx] = 1 - changed[test_idx]
    second = select_ttc_rows_ratio_adaptive(
        features=features,
        teacher_probs=probs,
        labels=changed,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        num_rows=5,
        ratio=0.001,
        policy="ratio_adaptive_core70",
        seed=4,
    )
    assert first.source_node_ids.tolist() == second.source_node_ids.tolist()
    assert torch.allclose(first.y_syn_soft, second.y_syn_soft)
    assert first.z_syn.shape[0] == 5
    assert first.diagnostics["condensed_nodes"] == 5
    assert first.diagnostics["mixup_virtual_count"] >= 0


def test_t32_prior_repair_reduces_selected_prior_kl() -> None:
    _, probs, *_ = _toy()
    selected = torch.tensor([0, 1, 7])
    teacher_prior = probs.mean(dim=0)
    before = compute_selected_soft_prior(probs[selected])
    repaired_ids = repair_soft_class_prior(selected, probs, teacher_prior, budget=4)
    after = compute_selected_soft_prior(probs[repaired_ids])
    assert repaired_ids.numel() == 4
    assert soft_prior_kl(after, teacher_prior) <= soft_prior_kl(before, teacher_prior) + 1e-8
