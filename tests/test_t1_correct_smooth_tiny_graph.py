import torch

from shadow_hgc.logits.correct_smooth import correct_and_smooth_probabilities


def test_t1_correct_smooth_uses_train_residual_and_destination_rows():
    logits = torch.zeros(3, 2)
    labels = torch.tensor([1, 0, 0])
    edge_index = torch.tensor([[0, 1], [1, 2]])

    result = correct_and_smooth_probabilities(
        logits=logits,
        labels=labels,
        train_idx=torch.tensor([0]),
        edge_index=edge_index,
        num_nodes=3,
        correct_alpha=1.0,
        correct_steps=1,
        smooth_alpha=0.0,
        smooth_steps=0,
    )

    probs = torch.softmax(result.logits, dim=1)
    assert probs[1, 1] > probs[1, 0]
    assert result.diagnostics["selection_uses_test"] is False
    assert result.diagnostics["normalization"] == "destination_row"
