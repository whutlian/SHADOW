from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class MissingBaseLogitsError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class BaseLogitCache:
    logits: torch.Tensor
    metadata: dict[str, Any]


def load_base_logit_cache(path: str | Path) -> BaseLogitCache:
    target = Path(path)
    if not target.exists():
        raise MissingBaseLogitsError("missing_base_logits")
    metadata: dict[str, Any] = {}
    if target.is_dir():
        meta_path = target / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for name in ("all_node_logits.pt", "logits.pt", "all_node_logits.npy", "logits.npy"):
            candidate = target / name
            if candidate.exists():
                return _load_logits_file(candidate, metadata)
        raise MissingBaseLogitsError("missing_base_logits")
    return _load_logits_file(target, metadata)


def _load_logits_file(path: Path, metadata: dict[str, Any]) -> BaseLogitCache:
    if path.suffix == ".npy":
        logits = torch.from_numpy(np.load(path)).to(torch.float32)
    elif path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            metadata = {**metadata, **dict(payload.get("metadata", {}))}
            payload = payload.get("logits", payload.get("all_node_logits", payload.get("probs")))
        if not isinstance(payload, torch.Tensor):
            raise ValueError(f"unsupported base logit payload in {path}")
        logits = payload.to(torch.float32)
    else:
        raise ValueError(f"unsupported base logit cache path: {path}")
    if logits.ndim != 2:
        raise ValueError("base logits must have shape [N, C]")
    metadata.setdefault("source_path", str(path))
    metadata.setdefault("uses_valid_labels_as_input", False)
    metadata.setdefault("uses_test_labels_as_input", False)
    return BaseLogitCache(logits=logits, metadata=metadata)


def find_base_logit_cache(base_logits_dir: str | Path, predictor: str) -> Path:
    root = Path(base_logits_dir)
    for suffix in (".pt", ".npy"):
        path = root / f"{predictor}_logits{suffix}"
        if path.exists():
            return path
    path = root / predictor
    if path.exists():
        return path
    return root / f"{predictor}_logits.pt"


def base_logit_metadata(
    *,
    dataset: str,
    base_predictor: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    feature_blocks: list[str],
    valid_acc: float | str = "",
    test_acc: float | str = "",
    macro_f1: float | str = "",
    predicted_classes: int | str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "base_predictor": base_predictor,
        "seed": int(seed),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "feature_blocks": list(feature_blocks),
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "predicted_classes": predicted_classes,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
    }
