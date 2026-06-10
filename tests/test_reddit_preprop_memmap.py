from pathlib import Path

import torch

from shadow_hgc.data.reddit import RedditGraphData, reddit_graph_spec
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank


def test_t24_reddit_tiny_preprop_memmap_no_e_by_d(tmp_path: Path):
    graph = RedditGraphData(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1]),
        train_idx=torch.tensor([0, 1]),
        valid_idx=torch.tensor([2]),
        test_idx=torch.tensor([3]),
        diagnostics={},
    )
    manifest = compute_preprop_filter_bank(
        dataset_name="reddit-toy",
        graph_spec=reddit_graph_spec(graph),
        feature_provider={graph.target_type: graph.x},
        target_node_ids=torch.arange(graph.num_nodes),
        train_target_ids=graph.train_idx,
        labels=graph.y,
        out_dir=tmp_path,
        blocks=("X0", "X1", "X2", "Xres1", "Y1", "structure"),
        feature_dim=2,
        edge_chunk_size=1,
    )
    assert manifest.uses_e_by_d_materialization is False
    assert (tmp_path / "manifest.json").exists()
