import json
from pathlib import Path

import numpy as np
import torch

from shadow_hgc.data.edge_stream import MemmapEdgeStream
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits, prepare_reddit_raw_memmaps
from shadow_hgc.preprop.chunked_spmm import chunked_destination_row_spmm
from shadow_hgc.preprop.streaming_spmm import streaming_destination_row_spmm


def test_memmap_edge_stream_reads_npy_chunks_without_stacked_edge_index(tmp_path: Path):
    np.save(tmp_path / "src.npy", np.array([0, 1, 2, 3, 4], dtype=np.int32))
    np.save(tmp_path / "dst.npy", np.array([1, 1, 2, 2, 0], dtype=np.int32))

    stream = MemmapEdgeStream(tmp_path / "src.npy", tmp_path / "dst.npy", chunk_size=2)
    chunks = list(stream)

    assert [chunk.src.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert [chunk.dst.tolist() for chunk in chunks] == [[1, 1], [2, 2], [0]]
    assert all(chunk.weight.dtype == torch.float32 for chunk in chunks)
    assert stream.num_edges == 5
    assert stream.storage_bytes == 40

    limited = MemmapEdgeStream(tmp_path / "src.npy", tmp_path / "dst.npy", chunk_size=2, edge_limit=3)
    assert [chunk.src.tolist() for chunk in limited] == [[0, 1], [2]]
    assert limited.num_edges == 3


def test_streaming_destination_row_spmm_matches_existing_chunked_spmm(tmp_path: Path):
    src = np.array([0, 1, 2, 0, 3, 4], dtype=np.int32)
    dst = np.array([1, 1, 2, 2, 2, 0], dtype=np.int32)
    np.save(tmp_path / "src.npy", src)
    np.save(tmp_path / "dst.npy", dst)
    source_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [1.0, 3.0],
        ]
    )

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    expected = chunked_destination_row_spmm(
        edge_index=edge_index,
        source_features=source_features,
        num_dst_nodes=3,
        edge_chunk_size=2,
    )
    actual = streaming_destination_row_spmm(
        edge_stream_factory=lambda: MemmapEdgeStream(tmp_path / "src.npy", tmp_path / "dst.npy", chunk_size=2),
        source_feature_getter=lambda ids: source_features[ids],
        feature_dim=2,
        num_dst_nodes=3,
        dst_rows=torch.arange(3),
    )

    assert torch.allclose(actual.block, expected.block, atol=1e-6)
    assert actual.diagnostics["full_edge_scans"] == 2
    assert actual.diagnostics["uses_e_by_d_materialization"] is False
    assert actual.diagnostics["materialized_stacked_edge_index"] is False


def test_prepare_reddit_raw_memmaps_from_npz_writes_streaming_manifest(tmp_path: Path):
    root = tmp_path / "Reddit"
    raw = root / "raw"
    raw.mkdir(parents=True)
    np.savez_compressed(
        raw / "reddit_graph.npz",
        row=np.array([0, 1, 2, 3], dtype=np.int32),
        col=np.array([1, 2, 3, 0], dtype=np.int32),
        data=np.ones(4, dtype=np.int64),
        shape=np.array([4, 4], dtype=np.int64),
        format=np.array(b"coo"),
    )
    np.savez_compressed(
        raw / "reddit_data.npz",
        feature=np.arange(12, dtype=np.float64).reshape(4, 3),
        node_types=np.array([1, 1, 2, 3], dtype=np.int32),
        node_ids=np.arange(4, dtype=np.int32),
        label=np.array([0, 1, 0, 1], dtype=np.int32),
    )

    manifest = prepare_reddit_raw_memmaps(root, out_dir=root / "processed" / "raw_memmap", overwrite=True)
    payload = json.loads((Path(manifest["memmap_root"]) / "manifest.json").read_text(encoding="utf-8"))

    assert payload["source"] == "reddit_raw_npz_streaming_extract"
    assert payload["num_nodes"] == 4
    assert payload["num_edges"] == 4
    assert payload["train_nodes"] == 2
    assert payload["valid_nodes"] == 1
    assert payload["test_nodes"] == 1
    assert np.load(Path(manifest["memmap_root"]) / payload["feature_path"], mmap_mode="r").dtype == np.float32
    assert np.load(Path(manifest["memmap_root"]) / payload["src_path"], mmap_mode="r").shape == (4,)

    labels, train, valid, test = load_reddit_raw_memmap_labels_and_splits(Path(manifest["memmap_root"]))
    assert labels.tolist() == [0, 1, 0, 1]
    assert train.tolist() == [0, 1]
    assert valid.tolist() == [2]
    assert test.tolist() == [3]
