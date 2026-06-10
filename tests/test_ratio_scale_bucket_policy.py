from shadow_hgc.ratio.scale_bucket import bucket_for_dataset, bucket_for_node_count, ratio_preset


def test_t24_scale_bucket_policy_is_deterministic():
    assert bucket_for_dataset("ogbn-arxiv") == "medium"
    assert bucket_for_dataset("Reddit") == "medium"
    assert bucket_for_dataset("ogbn-products") == "large"
    assert bucket_for_node_count(111_059_956) == "ultra"
    assert ratio_preset(dataset="ogbn-arxiv", preset="bucket_default") == [0.005]
    assert ratio_preset(dataset="ogbn-products", preset="bucket_sweep") == [0.0005, 0.001, 0.0025, 0.005]
