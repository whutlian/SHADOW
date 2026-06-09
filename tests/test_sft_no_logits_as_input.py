import pytest
import torch

from shadow_hgc.models.sft_teacher import SFTTableTeacher


def test_sft_teacher_rejects_logits_as_input_blocks():
    with pytest.raises(ValueError, match="logits"):
        SFTTableTeacher(
            {"self": 4, "logit_prop": 3},
            num_classes=3,
            model_type="sagn_lite",
        )


def test_sft_teacher_diagnostics_mark_no_logits_inputs():
    model = SFTTableTeacher({"self": 4, "typed:cite_ref": 4}, num_classes=3, model_type="sagn_lite")
    model.fit_block_stats(
        {"self": torch.randn(5, 4), "typed:cite_ref": torch.randn(5, 4)},
        train_rows=torch.tensor([0, 1, 2]),
    )
    assert model.diagnostics()["uses_logits_as_input"] is False
    assert model.diagnostics()["uses_teacher_logits"] is False
