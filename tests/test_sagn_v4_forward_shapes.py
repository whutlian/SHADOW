import torch

from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3


def test_t24_sagn_gamlp_v4_forward_shapes_and_raw_logits():
    blocks = {"self": torch.randn(4, 3), "Y1": torch.randn(4, 2)}
    for model_type in ["sagn_lite_v4", "gamlp_lite_v4"]:
        model = SFTTeacherV3({"self": 3, "Y1": 2}, num_classes=5, model_type=model_type, hidden_dim=8, norm="layernorm", block_dropout=0.05)
        model.fit_block_stats(blocks, train_rows=torch.tensor([0, 1]))
        logits = model(blocks)
        assert logits.shape == (4, 5)
        assert model.diagnostics()["model_type"] == model_type
        assert model.diagnostics()["final_logits_activation"] == "none"
