from __future__ import annotations

import torch

from shadow_hgc.train.train_sft_teacher import sft_loss


def test_sft_class_aware_losses_are_finite_and_training_only_alias_is_supported():
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [-0.5, 1.0, 0.0],
            [0.1, 0.2, 0.3],
            [1.5, 0.0, -2.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    train_labels = labels
    for loss_type in [
        "class_balanced_ce",
        "balanced_softmax",
        "logit_adjusted_ce_as_training_loss_only",
        "label_smoothing_ce",
        "focal_loss",
        "sqrt_weighted_ce",
    ]:
        loss = sft_loss(logits, labels, loss_type=loss_type, train_labels=train_labels, label_smoothing=0.05)
        assert torch.isfinite(loss)
        assert loss.ndim == 0
