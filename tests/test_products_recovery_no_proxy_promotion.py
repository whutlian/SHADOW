from shadow_hgc.sft.products_recovery import products_recovery_row, validate_products_recovery_row


def test_t24_products_proxy_or_bounded_rows_cannot_be_promoted():
    proxy = products_recovery_row(
        ratio=0.0025,
        method="P4_shadow_condensed_herding_b1",
        status="completed_proxy",
        accuracy=0.9,
        macro_f1=0.5,
        target_prototypes=10,
        shadow_nodes=5,
        condensed_edges=20,
        is_proxy=True,
        promotion_status="promoted",
    )
    assert proxy["promotion_status"] == "blocked_forbidden"
    bad = dict(proxy, is_proxy=False, uses_bounded_edges=True, promotion_status="promoted")
    assert validate_products_recovery_row(bad)["valid"] is False
