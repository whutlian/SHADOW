from __future__ import annotations

import pytest

from shadow_hgc.train.lazy_sft_memmap import _build_lazy_model


def _build(model_type: str):
    return _build_lazy_model(
        {"self": 4, "x1": 3},
        num_classes=2,
        model_type=model_type,
        hidden_dim=8,
        dropout=0.1,
        student_internal_style="gamlp_like",
        num_layers=2,
        block_dropout=0.0,
        hop_dropout=0.0,
        label_dropout=0.0,
        attention_heads=1,
        activation="relu",
        norm="none",
    )


def test_cross_student_table_heads_are_buildable() -> None:
    for model_type in ["linear_probe", "concat_mlp", "stt_gated_mixer", "sagn_lite_v2", "gamlp_lite_v2"]:
        model = _build(model_type)
        assert sum(param.numel() for param in model.parameters() if param.requires_grad) > 0
        assert model.diagnostics()["uses_logits_as_input"] is False


def test_cross_student_table_heads_reject_teacher_logit_blocks() -> None:
    with pytest.raises(ValueError, match="teacher/logit blocks are forbidden"):
        _build_lazy_model(
            {"self": 4, "teacher_logits": 2},
            num_classes=2,
            model_type="concat_mlp",
            hidden_dim=8,
            dropout=0.1,
            student_internal_style="concat_mlp",
            num_layers=2,
            block_dropout=0.0,
            hop_dropout=0.0,
            label_dropout=0.0,
            attention_heads=1,
            activation="relu",
            norm="none",
        )
