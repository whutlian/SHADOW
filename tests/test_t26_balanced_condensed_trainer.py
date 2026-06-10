import torch

from shadow_hgc.sft.balanced_condensed_trainer import balanced_batch_order, condensed_training_loss, within_class_sft_mixup


def test_balanced_batch_order_interleaves_classes():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    rows = torch.arange(labels.numel())

    order = balanced_batch_order(rows, labels, seed=1)
    prefix = labels[order[:3]].tolist()

    assert sorted(prefix) == [0, 1, 2]


def test_condensed_training_loss_supports_label_smoothing_and_logit_adjustment():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = condensed_training_loss(logits, labels, train_labels=labels, label_smoothing=0.05, logit_adjustment=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_within_class_mixup_never_crosses_labels():
    x = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
    y = torch.tensor([0, 0, 1, 1])

    mixed_x, mixed_y = within_class_sft_mixup(x, y, alpha=0.4, seed=4)

    assert mixed_x.shape == x.shape
    assert torch.equal(mixed_y, y)
    assert torch.all(mixed_x[:2] <= 2.0)
    assert torch.all(mixed_x[2:] >= 10.0)
