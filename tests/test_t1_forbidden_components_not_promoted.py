from scripts.run_t1_safe_cache_and_boost_stage import validate_safe_boost_row


def test_t1_forbidden_components_not_promoted():
    row = {"dataset": "toy", "uses_diffusion": True, "uses_bounded_edges": True, "promotion_status": "promoted"}

    checked = validate_safe_boost_row(row)

    assert checked["promotion_status"] == "invalid_for_promotion"
    assert checked["invalid_reasons"] == ["uses_diffusion", "uses_bounded_edges"]
