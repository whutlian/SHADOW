import torch

from shadow_hgc.sft.uca import coverage_gap_metrics, select_uca_labeled_nearest


def test_uca_selection_uses_unlabeled_features_but_not_unlabeled_labels():
    signatures = torch.tensor([[0.0], [0.1], [5.0], [5.2], [9.0], [9.1]])
    labels_a = torch.tensor([0, 0, 1, 1, 0, 1])
    labels_b = torch.tensor([0, 0, 1, 1, 1, 0])
    train_rows = torch.tensor([0, 2])
    target_rows = torch.arange(6)

    sel_a, stats_a = select_uca_labeled_nearest(signatures, labels_a, train_rows, target_rows, budget=2, num_domains=3, seed=7)
    sel_b, stats_b = select_uca_labeled_nearest(signatures, labels_b, train_rows, target_rows, budget=2, num_domains=3, seed=7)

    assert torch.equal(sel_a, sel_b)
    assert stats_a["uca_uses_valid_test_labels"] is False
    assert stats_b["domain_hist_all"] == stats_a["domain_hist_all"]


def test_coverage_gap_metrics_report_l1_l2_and_unsupported_domains():
    out = coverage_gap_metrics(torch.tensor([2, 0, 2]), torch.tensor([1, 0, 3]))

    assert out["coverage_gap_l1"] > 0
    assert out["coverage_gap_l2"] > 0
    assert out["domains_without_train_support"] == 1
