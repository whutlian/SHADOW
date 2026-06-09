import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.diagnostics.demand_equivalence import (
    compare_relation_demand_blocks,
    compute_destination_row_feature_demand,
)


def test_dblp_rplus_and_sfb_demand_match_under_same_edges_features_and_alpha():
    relation = DirectedRelation("paper", "written_by", "author")
    edge_index = torch.tensor([[0, 1, 2, 2], [0, 0, 1, 2]])
    paper_features = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    train_author_ids = torch.tensor([0, 1, 2])

    rplus = compute_destination_row_feature_demand(
        edge_index=edge_index,
        source_features=paper_features,
        num_target_nodes=3,
        target_rows=train_author_ids,
    )
    sfb = compute_destination_row_feature_demand(
        edge_index=edge_index,
        source_features=paper_features,
        num_target_nodes=3,
        target_rows=train_author_ids,
    )

    metrics = compare_relation_demand_blocks(
        dataset="dblp",
        relation_name="written_by",
        demand_a=rplus,
        demand_b=sfb,
        train_target_ids=train_author_ids,
        source_type=relation.source_type,
        destination_type=relation.destination_type,
        edge_direction_checked=True,
        alpha_normalization_checked=True,
    )

    assert metrics["cosine_mean"] >= 0.999
    assert metrics["row_l2_mean"] <= 1e-6
    assert metrics["allclose_fraction"] == 1.0


def test_dblp_demand_equivalence_detects_reversed_edge_direction():
    train_author_ids = torch.tensor([0, 1, 2])
    paper_features = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    correct_edges = torch.tensor([[0, 1, 2, 2], [0, 0, 1, 2]])
    reversed_edges = torch.stack([correct_edges[1], correct_edges[0]])

    correct = compute_destination_row_feature_demand(
        edge_index=correct_edges,
        source_features=paper_features,
        num_target_nodes=3,
        target_rows=train_author_ids,
    )
    wrong = compute_destination_row_feature_demand(
        edge_index=reversed_edges,
        source_features=paper_features,
        num_target_nodes=3,
        target_rows=train_author_ids,
    )
    metrics = compare_relation_demand_blocks(
        dataset="dblp",
        relation_name="written_by",
        demand_a=correct,
        demand_b=wrong,
        train_target_ids=train_author_ids,
    )

    assert metrics["row_l2_mean"] > 0.1
    assert metrics["allclose_fraction"] < 1.0


def test_dblp_demand_equivalence_detects_source_degree_normalization():
    train_author_ids = torch.tensor([0, 1, 2])
    paper_features = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    edge_index = torch.tensor([[0, 1, 2, 2], [0, 0, 1, 2]])
    correct = compute_destination_row_feature_demand(
        edge_index=edge_index,
        source_features=paper_features,
        num_target_nodes=3,
        target_rows=train_author_ids,
    )

    source_degree = torch.bincount(edge_index[0], minlength=paper_features.shape[0]).to(torch.float32)
    wrong = torch.zeros_like(correct)
    for src, dst in edge_index.t():
        wrong[dst] += paper_features[src] / source_degree[src].clamp_min(1.0)

    metrics = compare_relation_demand_blocks(
        dataset="dblp",
        relation_name="written_by",
        demand_a=correct,
        demand_b=wrong,
        train_target_ids=train_author_ids,
    )

    assert metrics["row_l2_mean"] > 0.1
    assert metrics["allclose_fraction"] < 1.0
