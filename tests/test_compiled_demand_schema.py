from __future__ import annotations

import json

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.compiled_table import (
    CompiledDemandBlock,
    compile_demand_table,
    schema_to_json,
)


def test_compiled_demand_schema_preserves_block_order_dims_and_total_dim():
    writes = DirectedRelation("author", "writes", "paper")
    cites = DirectedRelation("paper", "cite_ref", "paper")
    blocks = [
        (CompiledDemandBlock("self", "self", None, 3, "standardize"), torch.ones(4, 3)),
        (
            CompiledDemandBlock("demand:author:writes:paper", "feature_demand", writes, 2, "standardize"),
            torch.full((4, 2), 2.0),
        ),
        (
            CompiledDemandBlock("label_affinity:paper:cite_ref:paper", "label_affinity", cites, 4, "none"),
            torch.full((4, 4), 3.0),
        ),
        (CompiledDemandBlock("degree", "degree", None, 1, "standardize"), torch.zeros(4, 1)),
    ]

    table, schema = compile_demand_table(blocks)

    assert table.shape == (4, 10)
    assert schema.total_dim == 10
    assert [block.name for block in schema.blocks] == [
        "self",
        "demand:author:writes:paper",
        "label_affinity:paper:cite_ref:paper",
        "degree",
    ]
    assert [block.dim for block in schema.blocks] == [3, 2, 4, 1]

    encoded = json.loads(schema_to_json(schema))
    assert encoded["total_dim"] == 10
    assert [block["name"] for block in encoded["blocks"]] == [block.name for block in schema.blocks]
    assert encoded["blocks"][1]["relation"] == {
        "source_type": "author",
        "relation_name": "writes",
        "destination_type": "paper",
    }
