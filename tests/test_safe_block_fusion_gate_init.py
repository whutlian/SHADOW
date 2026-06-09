import torch

from shadow_hgc.models.safe_block_fusion import SafeBlockFusionClassifier


def test_non_self_gates_start_near_zero():
    model = SafeBlockFusionClassifier({"self": 4, "typed:noise": 3}, num_classes=2, hidden_dim=8)

    gates = model.gate_values()

    assert gates["typed:noise"] < 0.001
    assert gates["typed:noise"] > 0.0
