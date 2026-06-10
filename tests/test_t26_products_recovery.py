import torch

from shadow_hgc.sft.products_recovery_t26 import (
    compute_p0_recovery_diagnostics,
    mixed_class_budget,
    nearest_prototype_oracle,
    per_class_collapse_report,
)


def test_mixed_class_budget_uses_floor_and_exact_total():
    labels = torch.tensor([0] * 200 + [1] * 20 + [2] * 5)
    rows = torch.arange(labels.numel())

    budget = mixed_class_budget(labels, rows, total_budget=30, ratio=0.0005, num_classes=3, seed=42)

    assert sum(budget.values()) == 30
    assert all(value >= 1 for value in budget.values())
    assert budget[2] >= 1


def test_nearest_prototype_oracle_reports_accuracy_without_training():
    train_sig = torch.tensor([[0.0], [0.1], [3.0], [3.2]])
    train_labels = torch.tensor([0, 0, 1, 1])
    selected_pos = torch.tensor([0, 2])
    eval_sig = torch.tensor([[0.05], [3.1]])
    eval_labels = torch.tensor([0, 1])

    out = nearest_prototype_oracle(train_sig, train_labels, selected_pos, eval_sig, eval_labels, metric="euclidean")

    assert out["prototype_oracle_acc"] == 1.0
    assert out["centroid_oracle_acc"] == 1.0


def test_per_class_report_detects_predicted_collapse():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    selected = torch.tensor([0, 1, 2])
    pred = torch.tensor([0, 0, 0, 0, 1, 1])

    report = per_class_collapse_report(labels, selected, pred, num_classes=3)

    assert report[2]["selected_count"] == 0
    assert report[2]["predicted_count"] == 0
    assert report[2]["collapsed"] is True


def test_p0_diagnostics_encode_required_gates():
    diag = compute_p0_recovery_diagnostics(
        alltrain_acc=0.75,
        self_fit_acc=0.96,
        normalization_match=True,
        predicted_class_count=46,
        num_classes=47,
    )

    assert diag["p0a_passed"] is True
    assert diag["p0b_passed"] is True
    assert diag["p0f_normalization_parity"] is True
    assert diag["p0e_predicted_class_collapse"] is False
