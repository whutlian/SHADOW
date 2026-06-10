from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank


def test_filter_bank_reports_no_e_by_d_materialization(tmp_path):
    rel = DirectedRelation("paper", "cite_ref", "paper")
    manifest = compute_preprop_filter_bank(
        dataset_name="tiny",
        graph_spec={"target_type": "paper", "relations": {rel: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)}, "num_nodes": {"paper": 3}},
        feature_provider={"paper": torch.randn(3, 4)},
        target_node_ids=torch.arange(3),
        train_target_ids=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1, 0]),
        out_dir=tmp_path,
        blocks=("X0", "X1_cite_ref", "X2_cite_ref"),
        feature_dim=4,
        dtype="float16",
        edge_chunk_size=1,
    )

    assert manifest.uses_e_by_d_materialization is False
    for block in manifest.blocks:
        assert block.uses_e_by_d_materialization is False
        assert block.diagnostics.get("max_edge_chunk_size", 0) <= 1 or block.edge_scans == 0
