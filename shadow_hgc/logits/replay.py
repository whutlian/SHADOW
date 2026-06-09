from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.logits.io import load_logits_cache


def _weighted_f1(pred: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    total = 0.0
    score = 0.0
    for class_id in range(int(num_classes)):
        true_mask = labels == class_id
        pred_mask = pred == class_id
        support = float(true_mask.sum().item())
        if support == 0:
            continue
        tp = float((true_mask & pred_mask).sum().item())
        fp = float((~true_mask & pred_mask).sum().item())
        fn = float((true_mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        total += support
        score += support * f1
    return score / total if total else 0.0


def metrics_from_logits(logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor, *, num_classes: int | None = None) -> dict[str, Any]:
    z = logits if isinstance(logits, torch.Tensor) else torch.from_numpy(np.asarray(logits).copy())
    y = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(np.asarray(labels).copy())
    z = z.to(torch.float32)
    y = y.to(torch.long).flatten()
    if num_classes is None:
        num_classes = int(z.shape[1])
    pred = z.argmax(dim=1).to(torch.long)
    hist = torch.bincount(pred.clamp_min(0), minlength=int(num_classes)).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if hist.numel() else 0.0
    return {
        "accuracy": float((pred == y).to(torch.float32).mean().item()) if y.numel() else 0.0,
        "macro_f1": macro_f1_score(pred, y, num_classes=int(num_classes)),
        "weighted_f1": _weighted_f1(pred, y, int(num_classes)),
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
    }


def replay_logits_cache(cache_dir: str | Path, *, historical_test_acc: float | None = None, tolerance: float = 0.001) -> dict[str, Any]:
    loaded = load_logits_cache(cache_dir)
    if loaded.y_test is None:
        raise ValueError("cache has no test labels for replay")
    metrics = metrics_from_logits(loaded.test_logits, loaded.y_test, num_classes=loaded.meta.num_classes)
    historical = loaded.meta.accuracy if historical_test_acc is None else float(historical_test_acc)
    delta = "" if historical is None else float(metrics["accuracy"]) - float(historical)
    status = "available_verified"
    if historical is not None and abs(float(delta)) > float(tolerance):
        status = "invalid_replay_mismatch"
    return {
        "dataset": loaded.meta.dataset,
        "base_variant": loaded.meta.variant,
        "cache_path": str(Path(cache_dir)),
        "cache_status": status,
        "historical_test_acc": "" if historical is None else float(historical),
        "replay_test_acc": metrics["accuracy"],
        "delta_replay": delta,
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "predicted_class_count": metrics["predicted_class_count"],
        "prediction_entropy": metrics["prediction_entropy"],
        "train_nodes": int(loaded.train_idx.shape[0]),
        "valid_nodes": 0 if loaded.valid_idx is None else int(loaded.valid_idx.shape[0]),
        "test_nodes": 0 if loaded.test_idx is None else int(loaded.test_idx.shape[0]),
        "all_target_nodes": int(loaded.meta.num_target_nodes),
        "split_hash": loaded.meta.split_hash,
        "feature_hash": loaded.meta.feature_hash,
    }


def metadata_for_cache(cache_dir: str | Path) -> dict[str, Any]:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return payload["meta"]
