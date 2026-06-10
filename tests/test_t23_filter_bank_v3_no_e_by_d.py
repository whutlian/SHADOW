from pathlib import Path

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank_v3 import T23_ARXIV_FILTER_BANK_V3_BLOCKS, compute_t23_filter_bank_v3


def test_t23_filter_bank_v3_writes_blocks_without_e_by_d(tmp_path: Path):
    rel_ref = DirectedRelation("paper", "cite_ref", "paper")
    rel_by = DirectedRelation("paper", "cited_by", "paper")
    edge_ref = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_by = torch.stack([edge_ref[1], edge_ref[0]], dim=0)
    manifest = compute_t23_filter_bank_v3(
        dataset_name="toy-arxiv",
        graph_spec={"target_type": "paper", "relations": {rel_ref: edge_ref, rel_by: edge_by}},
        feature_provider={"paper": torch.randn(4, 5)},
        target_node_ids=torch.arange(4),
        train_target_ids=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1, 0, 1]),
        out_dir=tmp_path,
        blocks=(
            "X0",
            "X1_cite_ref",
            "X2_cite_ref",
            "X3_mix",
            "X4_mix",
            "Xres3_mix",
            "Y0_train_masked",
            "Y1_cite_ref",
            "Y4_mix",
            "Yres1_mix",
            "structure",
        ),
        feature_dim=4,
        edge_chunk_size=2,
    )
    names = {block.name for block in manifest.blocks}
    assert {"X4_mix", "Xres3_mix", "Y0_train_masked", "Y4_mix", "Yres1_mix"} <= names
    assert T23_ARXIV_FILTER_BANK_V3_BLOCKS[0] == "X0"
    assert manifest.uses_e_by_d_materialization is False
    assert all(block.uses_e_by_d_materialization is False for block in manifest.blocks)
