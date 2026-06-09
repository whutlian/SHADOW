import torch

from shadow_hgc.logits.pseudo_scap import build_t1_pseudo_labels


def test_t1_pseudo_scap_zeroes_low_confidence_nontrain_rows():
    logits = torch.tensor([[4.0, 0.0], [0.1, 0.0], [0.0, 4.0]])
    result = build_t1_pseudo_labels(
        logits,
        labels=torch.tensor([0, 1, 1]),
        train_idx=torch.tensor([0]),
        threshold=0.9,
        pseudo_weight=0.5,
        temperature=1.0,
    )

    assert result.weights.tolist() == [1.0, 0.0, 0.5]
    assert torch.allclose(result.pseudo[1], torch.zeros(2))
