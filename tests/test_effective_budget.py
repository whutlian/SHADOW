import pytest
import torch

from shadow_hgc.prototype.budgets import class_wise_budget
from shadow_hgc.prototype.cluster import class_wise_prototypes


def test_budget_upshifts_when_requested_below_class_minimum():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    train_idx = torch.arange(6)

    result = class_wise_budget(labels, train_idx, M_tau=2, min_proto_per_class=2, strict=False)

    assert result.requested_M_tau == 2
    assert result.effective_M_tau == 6
    assert result.budget_upshifted is True
    assert result.budgets == {0: 2, 1: 2, 2: 2}


def test_budget_strict_mode_rejects_impossible_minimum():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    train_idx = torch.arange(6)

    with pytest.raises(ValueError, match="requested M_tau"):
        class_wise_budget(labels, train_idx, M_tau=2, min_proto_per_class=2, strict=True)


def test_budget_respects_alpha_for_class_imbalance():
    labels = torch.tensor([0] * 16 + [1] * 4)
    train_idx = torch.arange(20)

    result = class_wise_budget(labels, train_idx, M_tau=6, min_proto_per_class=1, budget_alpha=0.5)

    assert result.effective_M_tau == 6
    assert result.budgets[0] == 4
    assert result.budgets[1] == 2


def test_class_wise_prototypes_reports_budget_metadata():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    train_idx = torch.arange(6)
    phi = torch.eye(6)

    result = class_wise_prototypes(
        phi_target=phi,
        signatures=phi,
        labels=labels,
        train_idx=train_idx,
        M_tau=2,
        min_proto_per_class=2,
        strict_budget=False,
        seed=0,
    )

    assert result.requested_M_tau == 2
    assert result.effective_M_tau == 6
    assert result.budget_upshifted is True
    assert result.prototype_features.shape[0] == 6
