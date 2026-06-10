import torch

from shadow_hgc.sft.coreset import class_sqrt_budget, select_classwise_coreset_rows


def test_class_sqrt_budget_is_exact_and_keeps_each_class():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    rows = torch.arange(6)

    budget = class_sqrt_budget(labels, rows, total=5)

    assert sum(budget.values()) == 5
    assert set(budget) == {0, 1, 2}
    assert all(value >= 1 for value in budget.values())


def test_select_classwise_coreset_rows_returns_train_rows_only():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    train_rows = torch.tensor([0, 1, 2, 3, 4, 5])
    signature = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ]
    )

    selected = select_classwise_coreset_rows(signature, labels, train_rows, total=4, mode="medoid", seed=7)

    assert selected.numel() == 4
    assert set(selected.tolist()).issubset(set(train_rows.tolist()))
    assert set(labels[selected].tolist()) == {0, 1, 2}
