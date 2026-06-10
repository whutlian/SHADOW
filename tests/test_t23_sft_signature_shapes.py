import pytest
import torch

from shadow_hgc.sft.signatures import build_sft_signature


def test_t23_sft_signature_shapes_and_forbids_logits():
    blocks = {"self": torch.randn(5, 4), "Y1": torch.randn(5, 3)}
    result = build_sft_signature(blocks, train_rows=torch.tensor([0, 1, 2]))
    assert result.signature.shape == (5, 7)
    assert result.block_dims == {"self": 4, "Y1": 3}
    assert result.uses_logits_as_input is False
    with pytest.raises(ValueError, match="logits"):
        build_sft_signature({"teacher_logits": torch.randn(5, 2)}, train_rows=torch.tensor([0, 1]))
