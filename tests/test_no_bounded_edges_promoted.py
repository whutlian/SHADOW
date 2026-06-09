from shadow_hgc.train.block_selection import validate_t2_promotion_row


def test_t2_promoted_rows_reject_bounded_edges():
    row = {
        "status": "promoted",
        "uses_logits_as_input": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": True,
        "uses_e_by_d_materialization": False,
    }

    result = validate_t2_promotion_row(row)

    assert result["valid"] is False
    assert "uses_bounded_edges" in result["invalid_reasons"]
