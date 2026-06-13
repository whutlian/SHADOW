from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_toy_papers100m_root(tmp_path: Path) -> Path:
    root = tmp_path / "toy_papers100m"
    root.mkdir(parents=True, exist_ok=True)
    node_feat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 2, -1, 1, -1], dtype=np.int64)
    edge_index = np.array(
        [
            [0, 2, 2, 4, 5, 1],
            [1, 1, 3, 1, 4, 4],
        ],
        dtype=np.int64,
    )
    train_idx = np.array([0, 1], dtype=np.int64)
    valid_idx = np.array([2], dtype=np.int64)
    test_idx = np.array([4], dtype=np.int64)
    np.save(root / "node_feat.npy", node_feat)
    np.save(root / "node_label.npy", labels)
    np.save(root / "edge_index.npy", edge_index)
    np.save(root / "train_idx.npy", train_idx)
    np.save(root / "valid_idx.npy", valid_idx)
    np.save(root / "test_idx.npy", test_idx)
    (root / "dataset_meta.json").write_text(
        json.dumps(
            {
                "dataset_name": "toy-papers100M",
                "num_nodes": int(node_feat.shape[0]),
                "num_edges": int(edge_index.shape[1]),
                "feature_dim": int(node_feat.shape[1]),
                "num_classes": 3,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def toy_expected_cite_ref_x1() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )


def toy_expected_cited_by_x1() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
