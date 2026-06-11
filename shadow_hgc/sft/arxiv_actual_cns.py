from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_hgc.sft.correct_smooth import correct_and_smooth


class MissingBaseLogitsError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ActualCNSResult:
    best_probs: torch.Tensor
    best_row: dict[str, Any]
    diagnostics: dict[str, Any]


def require_base_logits(logits_or_path: torch.Tensor | str | Path | None) -> torch.Tensor:
    if logits_or_path is None or str(logits_or_path) == "":
        raise MissingBaseLogitsError("missing_base_logits")
    if isinstance(logits_or_path, torch.Tensor):
        if logits_or_path.ndim != 2:
            raise ValueError("base logits must have shape [N, C]")
        return logits_or_path.to(torch.float32)
    path = Path(logits_or_path)
    if not path.exists():
        raise MissingBaseLogitsError("missing_base_logits")
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).to(torch.float32)
    if path.suffix == ".pt":
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded.get("logits", loaded.get("probs"))
        if not isinstance(loaded, torch.Tensor):
            raise ValueError(f"unsupported logits payload in {path}")
        return loaded.to(torch.float32)
    raise ValueError(f"unsupported base logits path: {path}")


def _metrics(probs: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, num_classes: int) -> dict[str, Any]:
    rows = rows.to(torch.long)
    if rows.numel() == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "predicted_classes": 0}
    pred = probs[rows].argmax(dim=1).to(torch.long).cpu()
    y = labels[rows].to(torch.long).cpu()
    encoded = y.clamp_min(0) * int(num_classes) + pred.clamp_min(0)
    confusion = torch.bincount(encoded, minlength=int(num_classes) * int(num_classes)).view(int(num_classes), int(num_classes))
    tp = torch.diag(confusion).to(torch.float64)
    fp = confusion.sum(dim=0).to(torch.float64) - tp
    fn = confusion.sum(dim=1).to(torch.float64) - tp
    macro = float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-12)).mean().item())
    return {
        "accuracy": float((pred == y).to(torch.float32).mean().item()),
        "macro_f1": macro,
        "predicted_classes": int(torch.unique(pred).numel()),
        "predicted_hist_json": json.dumps({str(i): int(v) for i, v in enumerate(torch.bincount(pred, minlength=int(num_classes)).tolist()) if int(v) > 0}, sort_keys=True),
    }


def run_actual_cns_grid(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    edge_index: torch.Tensor,
    num_classes: int,
    correction_alphas: list[float],
    smoothing_alphas: list[float],
    correction_steps: list[int],
    smoothing_steps: list[int],
    autoscale: bool = True,
) -> ActualCNSResult:
    base_logits = require_base_logits(logits)
    candidates: list[tuple[float, dict[str, Any], torch.Tensor]] = []
    for ca in correction_alphas:
        for sa in smoothing_alphas:
            for cs in correction_steps:
                for ss in smoothing_steps:
                    result = correct_and_smooth(
                        base_logits,
                        labels,
                        train_idx,
                        valid_idx,
                        test_idx,
                        edge_index,
                        num_classes=int(num_classes),
                        correction_alpha=float(ca),
                        smoothing_alpha=float(sa),
                        num_correction_steps=int(cs),
                        num_smoothing_steps=int(ss),
                        autoscale=bool(autoscale),
                    )
                    probs = result.logits_or_probs
                    valid = _metrics(probs, labels, valid_idx, int(num_classes))
                    test = _metrics(probs, labels, test_idx, int(num_classes))
                    row = {
                        "status": "completed_real",
                        "valid_acc": valid["accuracy"],
                        "valid_macro_f1": valid["macro_f1"],
                        "accuracy": test["accuracy"],
                        "macro_f1": test["macro_f1"],
                        "predicted_classes": test["predicted_classes"],
                        "cns_correction_alpha": float(ca),
                        "cns_smoothing_alpha": float(sa),
                        "cns_correction_steps": int(cs),
                        "cns_smoothing_steps": int(ss),
                        "cns_autoscale": bool(autoscale),
                    }
                    candidates.append((float(valid["accuracy"]), row, probs))
    if not candidates:
        raise ValueError("C&S grid must contain at least one candidate")
    candidates.sort(key=lambda item: (item[0], float(item[1].get("valid_macro_f1", 0.0))), reverse=True)
    best_valid, best_row, best_probs = candidates[0]
    return ActualCNSResult(
        best_probs=best_probs,
        best_row=best_row,
        diagnostics={
            "uses_cns_postprocess": True,
            "uses_valid_labels_for_selection": True,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "uses_test_labels_for_selection": False,
            "num_candidates": len(candidates),
            "best_valid_acc": best_valid,
        },
    )
