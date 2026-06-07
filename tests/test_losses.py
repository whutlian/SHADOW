import torch
import torch.nn.functional as F

from shadow_hgc.models.losses import prototype_cross_entropy


def test_cell_weighted_loss_matches_empirical_risk_formula():
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([2.0, 1.0, 3.0])

    loss = prototype_cross_entropy(logits, labels, weights, loss_type="weighted")

    ce = F.cross_entropy(logits, labels, reduction="none")
    expected = (weights * ce).sum() / weights.sum()
    assert torch.allclose(loss, expected)
