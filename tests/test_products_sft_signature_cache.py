from pathlib import Path

import torch

from shadow_hgc.sft.signature_cache import write_sft_signature_cache_from_blocks


def test_t24_products_sft_signature_cache_metadata(tmp_path: Path):
    result = write_sft_signature_cache_from_blocks(
        blocks={"X0": torch.randn(5, 3), "Y1": torch.randn(5, 2)},
        splits={"train": torch.tensor([0, 1, 2]), "valid": torch.tensor([3]), "test": torch.tensor([4])},
        train_rows=torch.tensor([0, 1, 2]),
        out_dir=tmp_path,
        dtype="float16",
    )
    assert (tmp_path / "train_signature.memmap").exists()
    assert result.metadata["block_names"] == ["X0", "Y1"]
    assert result.metadata["uses_logits"] is False
    assert result.metadata["uses_kd"] is False
