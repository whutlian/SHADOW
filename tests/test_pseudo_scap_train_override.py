import torch

from shadow_hgc.features.pseudo_scap import build_pseudo_labels


def test_pseudo_scap_train_nodes_override_with_one_hot_labels():
    logits = torch.tensor([[0.0, 4.0], [5.0, 0.0], [0.1, 0.0]])
    result = build_pseudo_labels(
        logits,
        labels=torch.tensor([0, 1, 0]),
        train_idx=torch.tensor([0, 1]),
        threshold=0.95,
        pseudo_weight=0.25,
        temperature=1.0,
    )

    assert torch.allclose(result.pseudo[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.pseudo[1], torch.tensor([0.0, 1.0]))
    assert result.weights[0].item() == 1.0
    assert result.weights[1].item() == 1.0
    assert result.diagnostics["train_override_count"] == 2
