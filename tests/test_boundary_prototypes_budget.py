import torch

from shadow_hgc.prototype.boundary import (
    boundary_aware_prototypes,
    score_boundary_nodes,
    split_boundary_budget,
)


def test_boundary_prototypes_match_effective_budget_when_feasible():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    train_idx = torch.arange(8)
    phi = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    signatures = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [3.0, 0.0],
            [3.2, 0.0],
            [0.0, 3.0],
            [0.2, 3.0],
            [3.0, 3.0],
            [3.2, 3.0],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [
            [6.0, 0.0],
            [0.2, 0.1],
            [4.0, 0.0],
            [0.3, 0.2],
            [0.0, 6.0],
            [0.1, 0.2],
            [0.0, 4.0],
            [0.2, 0.3],
        ],
        dtype=torch.float32,
    )

    result = boundary_aware_prototypes(
        phi_target=phi,
        signatures=signatures,
        labels=labels,
        train_idx=train_idx,
        logits=logits,
        M_tau=4,
        boundary_fraction=0.5,
        boundary_pool_fraction=0.5,
        clustering_method="kcenter",
        seed=7,
    )

    assert result.requested_M_tau == 4
    assert result.effective_M_tau == 4
    assert result.prototype_features.shape[0] == 4
    assert result.target_to_cell[train_idx].ge(0).all()
    assert torch.allclose(result.prototype_weights.sum(), torch.tensor(8.0))
    assert result.class_budget == {0: 2, 1: 2}


def test_boundary_split_respects_per_class_minimum_after_upshift():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    train_idx = torch.arange(6)
    phi = torch.eye(6, dtype=torch.float32)

    result = boundary_aware_prototypes(
        phi_target=phi,
        signatures=phi,
        labels=labels,
        train_idx=train_idx,
        probabilities=torch.full((6, 3), 1.0 / 3.0),
        M_tau=2,
        min_proto_per_class=2,
        boundary_fraction=0.5,
        clustering_method="kcenter",
        seed=0,
    )

    assert result.requested_M_tau == 2
    assert result.effective_M_tau == 6
    assert result.budget_upshifted is True
    assert result.class_budget == {0: 2, 1: 2, 2: 2}
    assert result.base_budget == {0: 1, 1: 1, 2: 1}
    assert result.boundary_budget == {0: 1, 1: 1, 2: 1}
    assert result.num_base_prototypes == 3
    assert result.num_boundary_prototypes == 3


def test_boundary_budget_counts_and_diagnostics_are_stable():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_idx = torch.arange(6)
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [0.0, 3.0],
            [0.0, 0.1],
            [0.0, 5.0],
        ],
        dtype=torch.float32,
    )

    scores = score_boundary_nodes(logits=logits, labels=labels, method="margin")
    split = split_boundary_budget({0: 3, 1: 3}, boundary_fraction=0.5)
    result = boundary_aware_prototypes(
        phi_target=torch.eye(6, dtype=torch.float32),
        signatures=torch.eye(6, dtype=torch.float32),
        labels=labels,
        train_idx=train_idx,
        boundary_scores=scores,
        M_tau=6,
        boundary_fraction=0.5,
        boundary_pool_fraction=0.5,
        clustering_method="kcenter",
        seed=3,
    )

    assert split.base_budget == {0: 1, 1: 1}
    assert split.boundary_budget == {0: 2, 1: 2}
    assert result.base_budget == {0: 1, 1: 1}
    assert result.boundary_budget == {0: 2, 1: 2}
    assert result.boundary_pool_size_by_class == {0: 2, 1: 2}
    assert result.num_base_prototypes == 2
    assert result.num_boundary_prototypes == 4
    assert result.boundary_score_stats["count"] == 6
    assert result.boundary_score_stats["method"] == "precomputed"
    assert result.boundary_score_stats["max"] > result.boundary_score_stats["min"]
