from __future__ import annotations

from shadow_hgc.eval.t22_promotion import validate_t22_promoted_row


def test_promoted_row_rejects_logits_kd_dense_p2_and_e_by_d():
    base = {
        "dataset": "ogbn-arxiv",
        "status": "promoted_short",
        "accuracy": 0.681,
        "macro_f1": 0.421,
        "predicted_class_count": 39,
        "uses_logits_as_input": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "full_edge_execution": True,
        "uses_memmap": True,
    }
    assert validate_t22_promoted_row(base)["valid"] is True
    bad = dict(base, uses_logits_as_input=True)
    result = validate_t22_promoted_row(bad)
    assert result["valid"] is False
    assert "uses_logits_as_input" in result["forbidden_flags"]
