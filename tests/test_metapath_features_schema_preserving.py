import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.metapath import metapath_target_features


def test_metapath_features_do_not_create_new_edge_types():
    relation = DirectedRelation("actor", "acts_in", "movie")
    edge_index = {relation: torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)}
    psi_target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    num_nodes = {"actor": 2, "movie": 2}

    result = metapath_target_features(
        edge_index=edge_index,
        relations=[relation],
        target_type="movie",
        psi_target=psi_target,
        num_nodes=num_nodes,
    )

    assert result.features.shape == (2, 2)
    assert result.path_names == ["movie-actor-movie"]
    assert set(result.exposed_relations) == {relation}
