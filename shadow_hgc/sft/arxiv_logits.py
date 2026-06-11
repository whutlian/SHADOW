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
        metadata = _load_cache_dir_metadata(target)
        for name in ("all_node_logits.pt", "logits.pt", "all_node_logits.npy", "logits.npy"):
            candidate = target / name
            if candidate.exists():
                return _load_logits_file(candidate, metadata)
        storage = metadata.get("_storage", {})
        all_target = storage.get("all_target_logits", {}) if isinstance(storage, dict) else {}
        if all_target:
            candidate = target / str(all_target.get("file", "all_target_logits.memmap"))
            if candidate.exists():
                return _load_memmap_logits(candidate, all_target, metadata)
        raise MissingBaseLogitsError("missing_base_logits")
    return _load_logits_file(target, metadata)


def _load_cache_dir_metadata(target: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    meta_path = target / "meta.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("meta"), dict):
            metadata.update(payload["meta"])
        if isinstance(payload.get("storage"), dict):
            metadata["_storage"] = payload["storage"]
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    return _normalize_metadata(metadata)


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    if "test_acc" not in normalized:
        if "accuracy" in normalized:
            normalized["test_acc"] = normalized["accuracy"]
        elif "base_accuracy" in normalized:
            normalized["test_acc"] = normalized["base_accuracy"]
    if "macro_f1" not in normalized and "base_macro_f1" in normalized:
        normalized["macro_f1"] = normalized["base_macro_f1"]
    if "predicted_classes" not in normalized:
        if "predicted_class_count" in normalized:
            normalized["predicted_classes"] = normalized["predicted_class_count"]
        elif "base_predicted_class_count" in normalized:
            normalized["predicted_classes"] = normalized["base_predicted_class_count"]
    normalized.setdefault("uses_valid_labels_as_input", False)
    normalized.setdefault("uses_test_labels_as_input", False)
    return normalized


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
    metadata = _normalize_metadata(metadata)
    return BaseLogitCache(logits=logits, metadata=metadata)


def _load_memmap_logits(path: Path, spec: dict[str, Any], metadata: dict[str, Any]) -> BaseLogitCache:
    shape = tuple(int(value) for value in spec.get("shape", []))
    if len(shape) != 2:
        raise ValueError(f"memmap logit cache must record shape [N, C]: {path}")
    dtype = np.dtype(str(spec.get("dtype", "float32")))
    mmap = np.memmap(path, mode="r", dtype=dtype, shape=shape)
    logits = torch.from_numpy(np.array(mmap, dtype=np.float32, copy=True))
    metadata = _normalize_metadata(metadata)
    metadata.setdefault("source_path", str(path))
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
