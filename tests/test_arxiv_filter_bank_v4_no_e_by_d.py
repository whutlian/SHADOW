from pathlib import Path

import pytest
import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank_v4 import compute_t24_filter_bank_v4, t24_arxiv_v4_blocks


def test_t24_arxiv_filter_bank_v4_no_e_by_d(tmp_path: Path):
    rel = DirectedRelation("paper", "cite_ref", "paper")
    manifest = compute_t24_filter_bank_v4(
        dataset_name="toy",
        graph_spec={"target_type": "paper", "relations": {rel: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}},
        feature_provider={"paper": torch.randn(3, 5)},
        target_node_ids=torch.arange(3),
        train_target_ids=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1, 0]),
        out_dir=tmp_path,
        blocks=("X0", "X1_cite_ref", "X2_cite_ref", "X3_mix", "X4_mix", "Xres2_mix", "Xres3_mix", "Y0_train_masked", "Y4_mix", "Yres1_mix", "structure"),
        feature_dim=4,
        edge_chunk_size=1,
    )
    assert manifest.uses_e_by_d_materialization is False
    assert "X4_mix" in {block.name for block in manifest.blocks}
    assert "Yres1_mix" in t24_arxiv_v4_blocks()
    with pytest.raises(ValueError, match="block_dim <= 128"):
        compute_t24_filter_bank_v4(
            dataset_name="toy",
            graph_spec={"target_type": "paper", "relations": {rel: torch.empty(2, 0, dtype=torch.long)}},
            feature_provider={"paper": torch.randn(3, 5)},
            target_node_ids=torch.arange(3),
            train_target_ids=torch.tensor([0]),
            labels=torch.tensor([0, 1, 0]),
            out_dir=tmp_path / "bad",
            feature_dim=256,
        )
