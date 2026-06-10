from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import gzip

from shadow_hgc.train.lazy_sft_memmap import LazyMemmapBlockStore, load_manifest_block_store, load_ogb_labels_and_splits


def _write_block(root: Path, name: str, values: np.ndarray) -> dict:
    path = root / f"block_{name}.memmap"
    mmap = np.memmap(path, mode="w+", dtype=np.float16, shape=values.shape)
    mmap[:] = values.astype(np.float16)
    mmap.flush()
    stats = {
        "block_name": name,
        "fit_scope": "train_target_rows",
        "frozen": True,
        "mean": values[:2].mean(axis=0).astype(float).tolist(),
        "std": np.maximum(values[:2].std(axis=0), 1e-6).astype(float).tolist(),
        "fit_rows": [0, 1],
    }
    (root / f"block_{name}_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return {"name": name, "path": path.name, "shape": list(values.shape), "dtype": "float16"}


def test_lazy_memmap_block_store_fetches_only_requested_rows(tmp_path):
    x0 = np.arange(24, dtype=np.float32).reshape(6, 4)
    x1 = (x0 + 100).astype(np.float32)
    metas = [_write_block(tmp_path, "X0", x0), _write_block(tmp_path, "X1_rel", x1)]
    store = LazyMemmapBlockStore.from_manifest(tmp_path, {"blocks": metas})

    batch = store.fetch(torch.tensor([1, 4, 5], dtype=torch.long), device=torch.device("cpu"))

    assert set(batch) == {"self", "x1_rel"}
    assert torch.allclose(batch["self"], torch.from_numpy(x0[[1, 4, 5]]).float(), atol=1e-3)
    assert store.block_dims == {"self": 4, "x1_rel": 4}
    assert store.max_batch_materialized_bytes <= 3 * 2 * 4 * 4


def test_load_manifest_block_store_uses_frozen_stats_without_full_tensor_load(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    meta = _write_block(tmp_path, "X0", values)
    manifest = {"dataset": "tiny", "blocks": [meta]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = load_manifest_block_store(tmp_path)

    assert store.stats["self"]["source"] == "train_target_rows"
    assert store.stats["self"]["frozen"] is True
    assert store.num_rows == 3


def test_load_ogb_labels_and_splits_reads_raw_gzip_without_processed_graph(tmp_path):
    root = tmp_path / "ogbn_arxiv"
    (root / "raw").mkdir(parents=True)
    (root / "split" / "time").mkdir(parents=True)
    for rel, values in {
        "raw/node-label.csv.gz": [4, 5, 6, 7],
        "split/time/train.csv.gz": [0, 1],
        "split/time/valid.csv.gz": [2],
        "split/time/test.csv.gz": [3],
    }.items():
        with gzip.open(root / rel, "wt", encoding="utf-8") as handle:
            for value in values:
                handle.write(f"{value}\n")

    labels, train, valid, test = load_ogb_labels_and_splits(root, split_name="time")

    assert labels.tolist() == [4, 5, 6, 7]
    assert train.tolist() == [0, 1]
    assert valid.tolist() == [2]
    assert test.tolist() == [3]
