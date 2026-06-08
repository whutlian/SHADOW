import pytest
import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.shadow_fusion import ShadowFusionClassifier


def test_shadow_fusion_uses_only_original_schema_relations():
    relation = DirectedRelation("author", "writes", "paper")
    model = ShadowFusionClassifier(
        in_channels={"author": 2, "paper": 2},
        out_channels=2,
        node_types=["author", "paper"],
        relations=[relation],
        target_type="paper",
        original_node_types=["author", "paper"],
        original_relations=[relation],
        hidden_dim=4,
        dropout=0.0,
    )

    out = model(
        {"author": torch.ones(1, 2), "paper": torch.zeros(1, 2)},
        {relation: torch.tensor([[0], [0]], dtype=torch.long)},
        {relation: torch.tensor([1.0])},
    )

    assert set(model.node_types) == {"author", "paper"}
    assert model.relations == [relation]
    assert set(out) == {"paper"}


def test_shadow_fusion_rejects_exposed_shadow_schema():
    original_relation = DirectedRelation("author", "writes", "paper")
    exposed_shadow_relation = DirectedRelation("author_shadow", "shadow_writes", "paper")

    with pytest.raises(ValueError, match="non-original node types"):
        ShadowFusionClassifier(
            in_channels={"author_shadow": 2, "paper": 2},
            out_channels=2,
            node_types=["author_shadow", "paper"],
            relations=[exposed_shadow_relation],
            target_type="paper",
            original_node_types=["author", "paper"],
            original_relations=[original_relation],
        )
