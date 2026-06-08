import torch
import torch.nn.functional as F

from shadow_hgc.models.losses import prototype_cross_entropy


def test_logit_adjusted_loss_matches_manual_calculation():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([4.0, 1.0, 1.0])
    class_prior = torch.tensor([0.25, 0.75])

    loss = prototype_cross_entropy(
        logits,
        labels,
        weights,
        loss_type="sqrt_weighted_logit_adjusted",
        class_prior=class_prior,
        logit_adjustment_tau=0.5,
    )

    adjusted = logits - 0.5 * torch.log(class_prior).unsqueeze(0)
    ce = F.cross_entropy(adjusted, labels, reduction="none")
    effective = torch.sqrt(weights)
    expected = (effective * ce).sum() / effective.sum()
    assert torch.allclose(loss, expected)
