from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.true_preprop import compute_preprop_blocks


def test_true_preprop_reports_no_e_by_d_and_respects_edge_chunk_size(tmp_path):
    relation = DirectedRelation("paper", "cited_by", "paper")
    edge_index = torch.tensor([[0, 1, 2, 0, 2], [1, 2, 0, 2, 1]], dtype=torch.long)
    x0 = torch.arange(12, dtype=torch.float32).view(3, 4)
    manifest = compute_preprop_blocks(
        dataset_name="tiny",
        target_type="paper",
        x_provider={"paper": x0, "train_rows": torch.tensor([0, 2], dtype=torch.long)},
        relations={relation: edge_index},
        output_dir=str(tmp_path),
        blocks=["X1"],
        feature_dim=4,
        dtype="float32",
        edge_chunk_size=2,
        dst_chunk_size=2,
        force_memmap=True,
        seed=42,
    )

    assert manifest.uses_e_by_d_materialization is False
    assert manifest.uses_dense_p2 is False
    assert manifest.uses_bounded_edges is False
    block = manifest.blocks[0]
    assert block.uses_e_by_d_materialization is False
    assert block.diagnostics["max_edge_chunk_size"] <= 2
    assert block.diagnostics["materialized_full_e_by_d"] is False
