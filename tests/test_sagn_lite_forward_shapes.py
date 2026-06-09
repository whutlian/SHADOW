import torch

from shadow_hgc.models.sft_teacher import SFTTableTeacher


def test_sagn_lite_forward_shapes_and_gate_logging():
    model = SFTTableTeacher(
        {"self": 4, "typed:cite_ref": 4, "structure": 3},
        num_classes=5,
        model_type="sagn_lite",
        hidden_dim=8,
        dropout=0.0,
    )
    blocks = {"self": torch.randn(6, 4), "typed:cite_ref": torch.randn(6, 4), "structure": torch.randn(6, 3)}
    model.fit_block_stats(blocks, train_rows=torch.tensor([0, 1, 2, 3]))
    logits = model(blocks)

    assert logits.shape == (6, 5)
    diag = model.diagnostics()
    assert diag["model_type"] == "sagn_lite"
    assert set(diag["block_gates"]) == {"self", "typed:cite_ref", "structure"}
    assert diag["final_logits_activation"] == "none"
