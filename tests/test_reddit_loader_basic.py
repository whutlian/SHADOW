from pathlib import Path

import torch

from shadow_hgc.data.reddit import load_reddit_dataset


def test_t24_reddit_loader_basic_from_processed_cache(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    data = {
        "x": torch.randn(4, 3),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "y": torch.tensor([0, 1, 0, 1]),
        "train_mask": torch.tensor([True, True, False, False]),
        "val_mask": torch.tensor([False, False, True, False]),
        "test_mask": torch.tensor([False, False, False, True]),
    }
    torch.save((data, None, object), processed / "data.pt")
    graph = load_reddit_dataset(tmp_path)
    assert graph.num_nodes == 4
    assert graph.num_edges == 3
    assert graph.train_idx.tolist() == [0, 1]
    assert graph.valid_idx.tolist() == [2]
    assert graph.test_idx.tolist() == [3]
