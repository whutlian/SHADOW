"""Prepare GraphSAINT-style data folders for external baselines.

The cloned baseline repositories do not agree on data layout.  This script
creates a lightweight compatibility root from the existing Shadow-HGC datasets:

  compat/
    reddit/{adj_full.npz, feats.npy, role.json, class_map.json}
    ogbn-arxiv/{adj_full.npz, feats.npy, role.json, class_map.json}
    ogbn-arxiv/raw -> file links used by GECC
    GraphSAINT/reddit -> ../reddit
    GraphSAINT/arxiv -> ../ogbn-arxiv
    ogbn_products -> ../ogbn_products

Only files under --output-root are created or replaced.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REQUIRED_GRAPHSAINT_FILES = ("adj_full.npz", "feats.npy", "role.json", "class_map.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create external-baseline data compatibility folders.")
    parser.add_argument("--source-root", type=Path, required=True, help="Existing Shadow-HGC dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Compatibility data root to create.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["reddit", "ogbn-arxiv"],
        choices=["reddit", "ogbn-arxiv"],
        help="GraphSAINT-style datasets to materialize.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        if dataset == "reddit":
            prepare_reddit(args.source_root, args.output_root / "reddit")
        elif dataset == "ogbn-arxiv":
            prepare_arxiv(args.source_root, args.output_root / "ogbn-arxiv")

    ensure_graphsaint_alias(args.output_root, "reddit", "reddit")
    ensure_graphsaint_alias(args.output_root, "arxiv", "ogbn-arxiv")
    ensure_raw_file_aliases(args.output_root / "reddit")
    ensure_raw_file_aliases(args.output_root / "ogbn-arxiv")
    ensure_products_alias(args.source_root, args.output_root)
    write_manifest(args.source_root, args.output_root)
    print(f"external baseline data root: {args.output_root}")
    return 0


def prepare_reddit(source_root: Path, out_dir: Path) -> None:
    raw_memmap = source_root / "Reddit" / "processed" / "raw_memmap"
    src_path = raw_memmap / "src.npy"
    dst_path = raw_memmap / "dst.npy"
    x_path = first_existing(raw_memmap / "x.float32.npy", raw_memmap / "feature.float64.npy")
    y_path = first_existing(raw_memmap / "y.int64.npy", raw_memmap / "label.int32.npy")
    train_path = raw_memmap / "train_idx.npy"
    val_path = raw_memmap / "valid_idx.npy"
    test_path = raw_memmap / "test_idx.npy"
    for path in (src_path, dst_path, x_path, y_path, train_path, val_path, test_path):
        require_file(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    features = np.load(x_path, mmap_mode="r")
    labels = np.asarray(np.load(y_path, mmap_mode="r")).reshape(-1)
    save_features(out_dir / "feats.npy", features)
    write_class_map(out_dir / "class_map.json", labels)
    write_role_json(out_dir / "role.json", np.load(train_path), np.load(val_path), np.load(test_path))
    write_sparse_adj(out_dir / "adj_full.npz", np.load(src_path, mmap_mode="r"), np.load(dst_path, mmap_mode="r"), features.shape[0])
    verify_graphsaint_dir(out_dir)


def prepare_arxiv(source_root: Path, out_dir: Path) -> None:
    arxiv_root = source_root / "ogbn_arxiv"
    raw = arxiv_root / "raw"
    split = arxiv_root / "split" / "time"
    for path in (
        raw / "edge.csv.gz",
        raw / "node-feat.csv.gz",
        raw / "node-label.csv.gz",
        split / "train.csv.gz",
        split / "valid.csv.gz",
        split / "test.csv.gz",
    ):
        require_file(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    features = read_csv_gz(raw / "node-feat.csv.gz", dtype=np.float32)
    labels = read_csv_gz(raw / "node-label.csv.gz", dtype=np.int64).reshape(-1)
    edges = read_csv_gz(raw / "edge.csv.gz", dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edge.csv.gz to have two columns, got shape {edges.shape}")
    save_features(out_dir / "feats.npy", features)
    write_class_map(out_dir / "class_map.json", labels)
    write_role_json(
        out_dir / "role.json",
        read_csv_gz(split / "train.csv.gz", dtype=np.int64).reshape(-1),
        read_csv_gz(split / "valid.csv.gz", dtype=np.int64).reshape(-1),
        read_csv_gz(split / "test.csv.gz", dtype=np.int64).reshape(-1),
    )
    write_sparse_adj(out_dir / "adj_full.npz", edges[:, 0], edges[:, 1], features.shape[0])
    verify_graphsaint_dir(out_dir)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these candidate files exists: " + ", ".join(str(path) for path in paths))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_csv_gz(path: Path, dtype: type[np.generic]) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return np.loadtxt(handle, delimiter=",", dtype=dtype)


def save_features(path: Path, features: np.ndarray) -> None:
    if path.exists():
        return
    np.save(path, np.asarray(features, dtype=np.float32))


def write_sparse_adj(path: Path, src: np.ndarray, dst: np.ndarray, num_nodes: int) -> None:
    if path.exists():
        return
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    if src.shape != dst.shape:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")
    data = np.ones(src.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    sp.save_npz(path, adj)


def write_role_json(path: Path, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if path.exists():
        return
    role = {
        "tr": [int(x) for x in np.asarray(train_idx).reshape(-1)],
        "va": [int(x) for x in np.asarray(val_idx).reshape(-1)],
        "te": [int(x) for x in np.asarray(test_idx).reshape(-1)],
    }
    path.write_text(json.dumps(role), encoding="utf-8")


def write_class_map(path: Path, labels: np.ndarray) -> None:
    if path.exists():
        return
    labels = np.asarray(labels).reshape(-1)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("{")
        for idx, label in enumerate(labels):
            if idx:
                handle.write(",")
            handle.write(json.dumps(str(idx)))
            handle.write(":")
            handle.write(json.dumps(int(label)))
        handle.write("}")


def ensure_graphsaint_alias(root: Path, alias: str, target_name: str) -> None:
    graphsaint = root / "GraphSAINT"
    graphsaint.mkdir(parents=True, exist_ok=True)
    ensure_dir_link(graphsaint / alias, Path("..") / target_name)


def ensure_raw_file_aliases(dataset_dir: Path) -> None:
    raw = dataset_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_GRAPHSAINT_FILES:
        ensure_file_link(raw / name, Path("..") / name)


def ensure_products_alias(source_root: Path, output_root: Path) -> None:
    products = source_root / "ogbn_products"
    if products.exists():
        ensure_dir_link(output_root / "ogbn_products", products)


def ensure_dir_link(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            return
    link_path.symlink_to(target, target_is_directory=True)


def ensure_file_link(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            return
    link_path.symlink_to(target)


def verify_graphsaint_dir(path: Path) -> None:
    missing = [name for name in REQUIRED_GRAPHSAINT_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing {missing}")


def write_manifest(source_root: Path, output_root: Path) -> None:
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "datasets": {
            "reddit": str(output_root / "reddit"),
            "ogbn-arxiv": str(output_root / "ogbn-arxiv"),
            "DeepCGC_arxiv": str(output_root / "GraphSAINT" / "arxiv"),
            "DeepCGC_reddit": str(output_root / "GraphSAINT" / "reddit"),
            "ogbn-products": str(output_root / "ogbn_products"),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
