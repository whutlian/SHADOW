import torch

from shadow_hgc.eval.metrics import macro_f1_score


def test_macro_f1_score_matches_simple_manual_case():
    pred = torch.tensor([0, 0, 1, 1])
    label = torch.tensor([0, 1, 1, 1])

    # class 0: precision 1/2, recall 1, f1 2/3
    # class 1: precision 1, recall 2/3, f1 4/5
    expected = ((2 / 3) + (4 / 5)) / 2
    assert abs(macro_f1_score(pred, label, num_classes=2) - expected) < 1e-6
