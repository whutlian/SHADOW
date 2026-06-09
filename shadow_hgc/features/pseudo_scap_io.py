from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shadow_hgc.features.pseudo_scap import PseudoTopK, dense_from_topk_sparse, dense_to_topk_sparse


def save_pseudo_scap(path: str | Path, block: torch.Tensor | PseudoTopK, *, metadata: dict | None = None) -> dict:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    if isinstance(block, PseudoTopK):
        np.save(root / "indices.npy", block.indices.detach().cpu().numpy())
        np.save(root / "values.npy", block.values.detach().cpu().numpy())
        if block.support is not None:
            np.save(root / "support.npy", block.support.detach().cpu().numpy())
        if block.entropy is not None:
            np.save(root / "entropy.npy", block.entropy.detach().cpu().numpy())
        if block.max_confidence is not None:
            np.save(root / "max_confidence.npy", block.max_confidence.detach().cpu().numpy())
        cache_bytes = sum(file.stat().st_size for file in root.glob("*.npy"))
        meta.update(block.metadata)
        meta.update({"num_rows": block.num_rows, "num_classes": block.num_classes, "cache_bytes": int(cache_bytes)})
    else:
        array = block.detach().cpu().numpy().astype(np.float32, copy=False)
        np.save(root / "dense.npy", array)
        meta.update({"dense_or_sparse": "dense", "num_rows": int(array.shape[0]), "num_classes": int(array.shape[1]), "cache_bytes": int((root / "dense.npy").stat().st_size)})
    meta.setdefault("block_type", "pseudo_scap")
    meta.setdefault("uses_train_labels_only", True)
    meta.setdefault("uses_validation_labels", False)
    meta.setdefault("uses_test_labels", False)
    (root / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_pseudo_scap(path: str | Path) -> tuple[torch.Tensor | PseudoTopK, dict]:
    root = Path(path)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("dense_or_sparse") == "sparse_topk":
        support = torch.from_numpy(np.load(root / "support.npy")) if (root / "support.npy").exists() else None
        entropy = torch.from_numpy(np.load(root / "entropy.npy")) if (root / "entropy.npy").exists() else None
        max_confidence = torch.from_numpy(np.load(root / "max_confidence.npy")) if (root / "max_confidence.npy").exists() else None
        block = PseudoTopK(
            indices=torch.from_numpy(np.load(root / "indices.npy")),
            values=torch.from_numpy(np.load(root / "values.npy")),
            support=support,
            entropy=entropy,
            max_confidence=max_confidence,
            num_rows=int(meta["num_rows"]),
            num_classes=int(meta["num_classes"]),
            metadata=meta,
        )
        return block, meta
    return torch.from_numpy(np.load(root / "dense.npy")), meta
