from shadow_hgc.ratio.scale_bucket import validate_t24_promoted_row


def test_t24_promoted_rows_reject_forbidden_components():
    clean = {"promotion_status": "promoted", "actual_full_node_ratio": 0.005, "uses_kd": False, "uses_dense_p2": False}
    assert validate_t24_promoted_row(clean)["valid"] is True
    bad = dict(clean, uses_kd=True, uses_coverage_medoid=True)
    result = validate_t24_promoted_row(bad)
    assert result["valid"] is False
    assert "uses_kd" in result["forbidden_flags"]
    assert "uses_coverage_medoid" in result["forbidden_flags"]
