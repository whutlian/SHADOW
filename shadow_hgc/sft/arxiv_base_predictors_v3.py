from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.sft.arxiv_logits import BaseLogitCache, load_base_logit_cache


REQUIRED_ARXIV_FILES: tuple[str, ...] = (
    "raw/node-feat.csv.gz",
    "raw/node-label.csv.gz",
    "raw/edge.csv.gz",
    "split/time/train.csv.gz",
    "split/time/valid.csv.gz",
    "split/time/test.csv.gz",
)


def _sha256_file(path: Path, limit_bytes: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = int(limit_bytes)
        while remaining > 0:
            chunk = handle.read(min(remaining, 65536))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()[:16]


def _load_gzip_ints(path: Path) -> torch.Tensor:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        arr = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    return torch.from_numpy(np.atleast_1d(arr).astype(np.int64, copy=False)).to(torch.long)


def validate_arxiv_split_and_feature_alignment(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    missing = [name for name in REQUIRED_ARXIV_FILES if not (root / name).exists()]
    if missing:
        return {"blocked": True, "failure_reason": "missing_arxiv_dataset", "missing_files": missing}
    try:
        labels = _load_gzip_ints(root / "raw" / "node-label.csv.gz")
        train = _load_gzip_ints(root / "split" / "time" / "train.csv.gz")
        valid = _load_gzip_ints(root / "split" / "time" / "valid.csv.gz")
        test = _load_gzip_ints(root / "split" / "time" / "test.csv.gz")
    except Exception as exc:  # pragma: no cover - defensive corruption path
        return {"blocked": True, "failure_reason": "invalid_arxiv_split_or_labels", "error": str(exc)}
    num_nodes = int(labels.numel())
    all_idx = torch.cat([train, valid, test])
    if all_idx.numel() == 0 or int(all_idx.min().item()) < 0 or int(all_idx.max().item()) >= num_nodes:
        return {"blocked": True, "failure_reason": "split_index_out_of_range", "num_nodes": num_nodes}
    split_hash_payload = {
        "train_n": int(train.numel()),
        "valid_n": int(valid.numel()),
        "test_n": int(test.numel()),
        "train_head": train[:16].tolist(),
        "valid_head": valid[:16].tolist(),
        "test_head": test[:16].tolist(),
    }
    split_hash = hashlib.sha256(json.dumps(split_hash_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    feature_manifest_hash = _sha256_file(root / "raw" / "node-feat.csv.gz")
    return {
        "blocked": False,
        "failure_reason": "",
        "num_nodes": num_nodes,
        "train_nodes": int(train.numel()),
        "valid_nodes": int(valid.numel()),
        "test_nodes": int(test.numel()),
        "split_hash": split_hash,
        "feature_manifest_hash": feature_manifest_hash,
    }


def load_validated_base_logits(path: str | Path, *, expected_nodes: int = 169_343, expected_classes: int = 40) -> BaseLogitCache:
    cache = load_base_logit_cache(path)
    if int(cache.logits.shape[0]) != int(expected_nodes) or int(cache.logits.shape[1]) != int(expected_classes):
        raise ValueError(
            f"base logits shape {tuple(cache.logits.shape)} does not match expected "
            f"({expected_nodes}, {expected_classes})"
        )
    return cache
