import torch

from shadow_hgc.prototype.sft_herding import select_sft_herding, sqrt_class_budget


def test_t23_sft_herding_uses_classwise_budget():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    train_rows = torch.arange(6)
    budget = sqrt_class_budget(labels, train_rows, total_budget=4)
    assert set(budget) == {0, 1, 2}
    assert sum(budget.values()) == 4
    result = select_sft_herding(
        signatures=torch.randn(6, 3),
        labels=labels,
        train_rows=train_rows,
        total_budget=4,
        mode="herding",
    )
    assert result.selected_rows.numel() == 4
    assert result.diagnostics["classwise_budget"] is True
    assert result.diagnostics["uses_validation_labels"] is False
