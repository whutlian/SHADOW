import torch

from shadow_hgc.logits.pseudo_scap import build_t1_pseudo_labels


def test_t1_pseudo_scap_train_labels_override_logits():
    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    result = build_t1_pseudo_labels(
        logits,
        labels=torch.tensor([0, 1]),
        train_idx=torch.tensor([0, 1]),
        threshold=0.95,
        pseudo_weight=0.25,
        temperature=1.0,
    )

    assert torch.allclose(result.pseudo[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.pseudo[1], torch.tensor([0.0, 1.0]))
    assert result.weights.tolist() == [1.0, 1.0]
