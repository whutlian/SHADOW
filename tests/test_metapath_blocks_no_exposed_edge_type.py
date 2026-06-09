from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation, ensure_schema_preserved
from shadow_hgc.features.metapath_blocks import compute_metapath_feature_blocks


def test_metapath_blocks_do_not_expose_synthetic_edge_types():
    actor_movie = DirectedRelation("actor", "acts_in", "movie")
    result = compute_metapath_feature_blocks(
        edge_index={actor_movie: torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)},
        relations=[actor_movie],
        target_type="movie",
        target_features=torch.eye(2),
        num_nodes={"movie": 2, "actor": 2},
        requested_blocks=["MAM"],
    )

    assert result.exposed_relations == [actor_movie]
    assert ensure_schema_preserved(
        exposed_node_types=["actor", "movie"],
        exposed_relations=result.exposed_relations,
        original_node_types=["actor", "movie"],
        original_relations=[actor_movie],
    )
