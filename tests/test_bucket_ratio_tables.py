from shadow_hgc.ratio.scale_bucket import fixed_bucket_main_rows


def test_t24_bucket_ratio_table_main_ratios():
    rows = {row["dataset"]: row for row in fixed_bucket_main_rows()}
    assert rows["ogbn-arxiv"]["main_ratio"] == 0.005
    assert rows["Reddit"]["main_ratio"] == 0.005
    assert rows["ogbn-products"]["main_ratio"] == 0.0025
    assert rows["ogbn-products"]["ratio_mode"] == "full_node"
