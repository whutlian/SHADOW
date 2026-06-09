import torch

from tests.test_imdb_relation_inventory import _imdb_toy_graph
from shadow_hgc.diagnostics.imdb_inventory import compare_imdb_clean_s1_to_sfb_metapaths


def test_imdb_clean_s1_and_sfb_metapath_blocks_are_identical_when_reused():
    metrics = compare_imdb_clean_s1_to_sfb_metapaths(_imdb_toy_graph(), target_rows=torch.tensor([0, 1, 2]))

    assert set(metrics) == {"MAM", "MDM", "MKM"}
    for block_metrics in metrics.values():
        assert block_metrics["cosine_mean"] >= 0.999
        assert block_metrics["row_l2_mean"] <= 1e-6
        assert block_metrics["allclose_fraction"] == 1.0
