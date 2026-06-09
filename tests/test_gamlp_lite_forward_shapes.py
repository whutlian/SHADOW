import torch

from shadow_hgc.models.sft_teacher import SFTTableTeacher


def test_gamlp_lite_forward_shapes_and_gate_logging():
    model = SFTTableTeacher(
        {"self": 4, "X1": 4, "lad:cite_ref": 2},
        num_classes=3,
        model_type="gamlp_lite",
        hidden_dim=8,
        dropout=0.0,
    )
    blocks = {"self": torch.randn(7, 4), "X1": torch.randn(7, 4), "lad:cite_ref": torch.randn(7, 2)}
    model.fit_block_stats(blocks, train_rows=torch.tensor([0, 2, 4]))
    logits = model(blocks)

    assert logits.shape == (7, 3)
    diag = model.diagnostics()
    assert diag["model_type"] == "gamlp_lite"
    assert set(diag["block_gates"]) == {"X1", "lad:cite_ref"}
    assert diag["final_logits_activation"] == "none"
