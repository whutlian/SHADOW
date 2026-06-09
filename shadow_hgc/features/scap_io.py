from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shadow_hgc.features.scap import SparseTopKSCAP


def write_scap_memmap(
    path: str | Path,
    block: torch.Tensor | SparseTopKSCAP,
    *,
    metadata: dict,
) -> dict:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    if isinstance(block, SparseTopKSCAP):
        np.save(root / "class_ids.npy", block.class_ids.detach().cpu().numpy())
        np.save(root / "values.npy", block.values.detach().cpu().numpy())
        cache_bytes = int((root / "class_ids.npy").stat().st_size + (root / "values.npy").stat().st_size)
        meta.update(block.metadata)
        meta.update({"num_target_rows": block.num_rows, "num_classes": block.num_classes, "cache_bytes": cache_bytes})
    else:
        array = block.detach().cpu().numpy()
        np.save(root / "dense.npy", array)
        meta.update({"dense_or_sparse": "dense", "num_target_rows": int(array.shape[0]), "num_classes": int(array.shape[1]), "cache_bytes": int((root / "dense.npy").stat().st_size)})
    meta.setdefault("block_type", "scap")
    meta.setdefault("uses_train_labels_only", True)
    meta.setdefault("uses_validation_labels", False)
    meta.setdefault("uses_test_labels", False)
    (root / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def read_scap_memmap(path: str | Path) -> tuple[torch.Tensor | SparseTopKSCAP, dict]:
    root = Path(path)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("dense_or_sparse") == "sparse_topk":
        class_ids = torch.from_numpy(np.load(root / "class_ids.npy", mmap_mode="r").copy())
        values = torch.from_numpy(np.load(root / "values.npy", mmap_mode="r").copy())
        return SparseTopKSCAP(
            class_ids=class_ids,
            values=values,
            num_rows=int(meta["num_target_rows"]),
            num_classes=int(meta["num_classes"]),
            metadata=meta,
        ), meta
    dense = torch.from_numpy(np.load(root / "dense.npy", mmap_mode="r").copy())
    return dense, meta
