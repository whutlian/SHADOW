from __future__ import annotations

import pytest

from shadow_hgc.preprop.resource import validate_products_full_execution_row


def test_products_full_execution_rejects_bounded_edges_as_promoted_performance():
    row = {
        "dataset": "ogbn-products",
        "status": "promoted",
        "uses_bounded_edges": True,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_e_by_d_materialization": False,
    }
    with pytest.raises(ValueError, match="bounded"):
        validate_products_full_execution_row(row)
