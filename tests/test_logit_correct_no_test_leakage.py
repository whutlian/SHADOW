import torch

from shadow_hgc.logits.correct_lite import correct_error_then_smooth


def test_logit_correct_uses_train_labels_only():
    logits = torch.zeros(4, 2)
    labels = torch.tensor([1, 1, 0, 0])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    result_a = correct_error_then_smooth(
        logits=logits,
        labels=labels,
        train_idx=torch.tensor([0]),
        edge_index=edge_index,
        num_nodes=4,
        correct_steps=1,
        correct_alpha=0.5,
        beta=1.0,
        smooth_steps=1,
        smooth_alpha=0.5,
    )
    labels_with_changed_test = torch.tensor([1, 1, 1, 1])
    result_b = correct_error_then_smooth(
        logits=logits,
        labels=labels_with_changed_test,
        train_idx=torch.tensor([0]),
        edge_index=edge_index,
        num_nodes=4,
        correct_steps=1,
        correct_alpha=0.5,
        beta=1.0,
        smooth_steps=1,
        smooth_alpha=0.5,
    )

    assert torch.allclose(result_a.logits, result_b.logits)
    assert result_a.diagnostics["uses_test_labels"] is False
    assert result_a.diagnostics["uses_validation_labels"] is False
