import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.diagnostics.imdb_inventory import audit_imdb_relation_inventory


def _imdb_toy_graph() -> HeteroGraphData:
    directs = DirectedRelation("director", "directs", "movie")
    acts_in = DirectedRelation("actor", "acts_in", "movie")
    keyword_in = DirectedRelation("keyword", "keyword_in", "movie")
    has_actor = DirectedRelation("movie", "has_actor", "actor")
    directed_by = DirectedRelation("movie", "directed_by", "director")
    has_keyword = DirectedRelation("movie", "has_keyword", "keyword")
    return HeteroGraphData(
        dataset_name="imdb",
        target_type="movie",
        node_features={
            "movie": torch.randn(3, 4),
            "director": torch.randn(2, 3),
            "actor": torch.randn(2, 3),
            "keyword": torch.randn(2, 2),
        },
        edge_index={
            directs: torch.tensor([[0, 1], [0, 1]]),
            acts_in: torch.tensor([[0, 1], [0, 2]]),
            keyword_in: torch.tensor([[0, 1], [1, 2]]),
            has_actor: torch.tensor([[0, 2], [0, 1]]),
            directed_by: torch.tensor([[0, 1], [0, 1]]),
            has_keyword: torch.tensor([[1, 2], [0, 1]]),
        },
        labels=torch.tensor([0, 1, 2]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.empty(0, dtype=torch.long),
        test_idx=torch.tensor([2]),
        relations=[directs, acts_in, keyword_in, has_actor, directed_by, has_keyword],
        num_nodes={"movie": 3, "director": 2, "actor": 2, "keyword": 2},
    )


def test_imdb_relation_inventory_includes_keyword_signal_and_all_clean_paths():
    audit = audit_imdb_relation_inventory(_imdb_toy_graph())

    assert audit["typed:directs_exists"] is True
    assert audit["typed:acts_in_exists"] is True
    assert audit["typed:keyword_in_exists"] is True
    assert audit["MAM_available"] is True
    assert audit["MDM_available"] is True
    assert audit["MKM_available"] is True
    assert "keyword->keyword_in->movie" in audit["incoming_target_relations"]
