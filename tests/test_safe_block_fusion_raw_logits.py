import torch

from shadow_hgc.models.safe_block_fusion import SafeBlockFusionClassifier


def test_safe_block_fusion_outputs_raw_logits_that_may_be_negative():
    model = SafeBlockFusionClassifier({"self": 2, "useful": 2}, num_classes=2, hidden_dim=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.self_head[-1].bias.copy_(torch.tensor([-1.5, 0.5]))

    logits = model({"self": torch.ones(3, 2), "useful": torch.ones(3, 2)})

    assert torch.any(logits < 0)
    assert model.diagnostics()["final_logits_activation"] == "none"
