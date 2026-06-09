import torch

from shadow_hgc.features.pseudo_scap import build_pseudo_labels


def test_pseudo_scap_confidence_gate_masks_low_confidence_nontrain_nodes():
    logits = torch.tensor([[3.0, 0.0], [0.1, 0.0], [0.0, 4.0]])
    result = build_pseudo_labels(
        logits,
        labels=torch.tensor([0, 1, 1]),
        train_idx=torch.tensor([0]),
        threshold=0.9,
        pseudo_weight=0.5,
        temperature=1.0,
    )

    assert result.weights[0].item() == 1.0
    assert result.weights[1].item() == 0.0
    assert result.weights[2].item() == 0.5
    assert result.diagnostics["nontrain_used_count"] == 1
