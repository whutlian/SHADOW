from scripts.run_t1_logit_affinity_stage import validate_t1_promotion_row


def test_t1_no_forbidden_promotion_rejects_bounded_edges_and_dense_p2():
    row = {
        "dataset": "toy",
        "accuracy": 0.8,
        "uses_diffusion": False,
        "uses_dense_p2": True,
        "uses_bounded_edges": True,
    }

    result = validate_t1_promotion_row(row)

    assert result["valid_for_promotion"] is False
    assert result["promotion_status"] == "invalid_for_promotion"
    assert result["invalid_reasons"] == ["uses_dense_p2", "uses_bounded_edges"]
