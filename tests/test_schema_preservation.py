import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.graph.materialize import RelationShadowPlan, materialize_condensed_graph


def test_materialized_shadow_graph_exposes_only_original_schema():
    target_type = "paper"
    relations = [
        DirectedRelation("author", "writes", "paper"),
    ]
    prototype_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([3.0, 1.0])
    shadows = {
        relations[0]: RelationShadowPlan(
            shadow_features=torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
            assignment=torch.tensor([0, 1]),
            skeleton_edge_index=None,
            skeleton_edge_weight=None,
        )
    }

    condensed = materialize_condensed_graph(
        target_type=target_type,
        original_node_types={"paper", "author"},
        original_relations=set(relations),
        prototype_features=prototype_x,
        prototype_labels=labels,
        prototype_weights=weights,
        relation_plans=shadows,
    )

    assert set(condensed.node_features) == {"paper", "author"}
    assert set(condensed.edge_index) == set(relations)
    assert "author_shadow" not in condensed.node_features
    assert torch.equal(condensed.target_indices, torch.tensor([0, 1]))
