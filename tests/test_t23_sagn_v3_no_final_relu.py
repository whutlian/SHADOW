import torch

from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3


def test_t23_sagn_and_gamlp_v3_return_raw_logits_without_final_relu():
    blocks = {"self": torch.randn(6, 4), "Y1_cite_ref": torch.randn(6, 3)}
    for model_type in ["sagn_lite_v3", "gamlp_lite_v3"]:
        model = SFTTeacherV3(
            {"self": 4, "Y1_cite_ref": 3},
            num_classes=2,
            model_type=model_type,  # type: ignore[arg-type]
            hidden_dim=8,
            label_dropout=0.1,
            attention_heads=2,
        )
        model.fit_block_stats(blocks, train_rows=torch.arange(4))
        with torch.no_grad():
            model.classifier.bias.fill_(-5.0)
        logits = model(blocks)
        assert logits.shape == (6, 2)
        assert (logits < 0).any()
        diag = model.diagnostics()
        assert diag["model_type"] == model_type
        assert diag["final_logits_activation"] == "none"
        assert diag["label_dropout"] == 0.1
        assert diag["attention_heads"] == 2
