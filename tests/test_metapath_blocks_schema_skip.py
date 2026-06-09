from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.metapath_blocks import compute_metapath_feature_blocks


def test_metapath_blocks_skip_requested_paths_missing_from_schema():
    author_paper = DirectedRelation("author", "writes", "paper")
    result = compute_metapath_feature_blocks(
        edge_index={author_paper: torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)},
        relations=[author_paper],
        target_type="paper",
        target_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        num_nodes={"paper": 2, "author": 2},
        requested_blocks=["PAP", "PSP"],
    )

    assert list(result.blocks) == ["PAP"]
    assert result.skipped_blocks == ["PSP"]
    assert result.blocks["PAP"].shape == (2, 2)
