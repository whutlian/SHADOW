from __future__ import annotations

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.compiled_table import (
    CompiledDemandBlock,
    compile_demand_table,
    schema_to_json,
)


def test_prototype_and_original_target_rows_compile_to_identical_schema():
    relation = DirectedRelation("author", "writes", "paper")
    prototype_blocks = [
        (CompiledDemandBlock("self", "self", None, 2, "standardize"), torch.randn(2, 2)),
        (
            CompiledDemandBlock("demand:author:writes:paper", "feature_demand", relation, 3, "standardize"),
            torch.randn(2, 3),
        ),
        (CompiledDemandBlock("degree", "degree", None, 4, "standardize"), torch.randn(2, 4)),
    ]
    original_target_blocks = [
        (block, torch.randn(5, block.dim))
        for block, _ in prototype_blocks
    ]

    prototype_table, prototype_schema = compile_demand_table(prototype_blocks)
    original_table, original_schema = compile_demand_table(original_target_blocks)

    assert prototype_table.shape == (2, prototype_schema.total_dim)
    assert original_table.shape == (5, original_schema.total_dim)
    assert prototype_schema == original_schema
    assert schema_to_json(prototype_schema) == schema_to_json(original_schema)
