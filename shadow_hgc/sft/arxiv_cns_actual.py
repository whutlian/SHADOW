from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shadow_hgc.sft.arxiv_actual_cns import ActualCNSResult, run_actual_cns_grid
from shadow_hgc.sft.arxiv_logits import BaseLogitCache, MissingBaseLogitsError, load_base_logit_cache


@dataclass(frozen=True)
class T30CNSResult:
    best_probs: torch.Tensor
    best_row: dict[str, Any]
    diagnostics: dict[str, Any]


def run_t30_cns_grid(
    *,
    cache: BaseLogitCache,
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
) -> T30CNSResult:
    if cache.metadata.get("uses_valid_labels_as_input") or cache.metadata.get("uses_test_labels_as_input"):
        raise ValueError("base predictor cache uses forbidden validation/test labels as input")
    result: ActualCNSResult = run_actual_cns_grid(
        logits=cache.logits,
        labels=labels,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        edge_index=edge_index,
        num_classes=int(num_classes),
        correction_alphas=correction_alphas,
        smoothing_alphas=smoothing_alphas,
        correction_steps=correction_steps,
        smoothing_steps=smoothing_steps,
    )
    best = dict(result.best_row)
    best["status"] = "completed_long"
    best["cns_valid_acc"] = best.get("valid_acc", "")
    best["cns_test_acc"] = best.get("accuracy", "")
    diagnostics = {
        **result.diagnostics,
        "uses_cns_postprocess": True,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "base_valid_acc": cache.metadata.get("valid_acc", ""),
        "base_test_acc": cache.metadata.get("test_acc", ""),
    }
    return T30CNSResult(best_probs=result.best_probs, best_row=best, diagnostics=diagnostics)


__all__ = ["BaseLogitCache", "MissingBaseLogitsError", "load_base_logit_cache", "run_t30_cns_grid"]
