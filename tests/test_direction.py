import pytest
import torch

from shadow_hgc.data.schemas import (
    DirectedRelation,
    ensure_schema_preserved,
    validate_edge_index_direction,
)


def test_edge_index_zero_row_is_source_and_one_row_is_destination():
    relation = DirectedRelation("author", "writes", "paper")
    edge_index = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)

    validate_edge_index_direction(relation, edge_index, num_src_nodes=3, num_dst_nodes=3)


def test_direction_checker_rejects_out_of_range_source_or_destination():
    relation = DirectedRelation("author", "writes", "paper")

    bad_source = torch.tensor([[0, 3], [0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="source"):
        validate_edge_index_direction(relation, bad_source, num_src_nodes=3, num_dst_nodes=2)

    bad_destination = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="destination"):
        validate_edge_index_direction(relation, bad_destination, num_src_nodes=2, num_dst_nodes=2)


def test_schema_preservation_uses_only_original_types_and_relations():
    original_relations = {
        DirectedRelation("paper", "cite_ref", "paper"),
        DirectedRelation("paper", "cited_by", "paper"),
        DirectedRelation("author", "writes", "paper"),
    }

    assert ensure_schema_preserved(
        exposed_node_types={"paper", "author"},
        exposed_relations=original_relations,
        original_node_types={"paper", "author"},
        original_relations=original_relations,
    )

    with pytest.raises(ValueError, match="node types"):
        ensure_schema_preserved(
            exposed_node_types={"paper", "author_shadow"},
            exposed_relations=original_relations,
            original_node_types={"paper", "author"},
            original_relations=original_relations,
        )
