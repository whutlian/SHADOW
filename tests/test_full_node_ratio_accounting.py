from shadow_hgc.ratio.scale_bucket import account_full_node_ratio


def test_t24_full_node_ratio_accounting_uses_total_condensed_nodes():
    row = account_full_node_ratio(original_total_nodes=1000, target_prototypes=3, shadow_nodes=2, other_condensed_nodes=5, condensed_edges=20)
    assert row["total_condensed_nodes"] == 10
    assert row["actual_full_node_ratio"] == 0.01
    assert row["total_condensed_edges"] == 20
