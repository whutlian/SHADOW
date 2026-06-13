from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _as_probs(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().float()
    row_sum = x.sum(dim=1, keepdim=True)
    if bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)):
        return x / row_sum.clamp_min(1e-12)
    return torch.softmax(x, dim=1)


def _k_from_mode(mode: str) -> int:
    digits = "".join(ch for ch in str(mode) if ch.isdigit())
    return int(digits) if digits else 8


@dataclass(frozen=True)
class STTTeacherCache:
    mode: str
    dense_probs: torch.Tensor | None = None
    topk_class_ids: torch.Tensor | None = None
    topk_probs: torch.Tensor | None = None
    tail_mass: torch.Tensor | None = None
    tail_prior: torch.Tensor | None = None

    def reconstruct_rows(self, rows: torch.Tensor, *, num_classes: int) -> torch.Tensor:
        rows = rows.detach().cpu().long()
        if self.dense_probs is not None:
            return self.dense_probs[rows].float()
        assert self.topk_class_ids is not None and self.topk_probs is not None and self.tail_mass is not None
        out = torch.zeros((rows.numel(), int(num_classes)), dtype=torch.float32)
        ids = self.topk_class_ids[rows].long()
        vals = self.topk_probs[rows].float()
        batch = torch.arange(rows.numel()).view(-1, 1).expand_as(ids)
        out[batch, ids] = vals
        tail = self.tail_mass[rows].float()
        if self.tail_prior is None:
            tail_prior = torch.ones(int(num_classes), dtype=torch.float32) / float(num_classes)
        else:
            tail_prior = self.tail_prior.float()
            tail_prior = tail_prior / tail_prior.sum().clamp_min(1e-12)
        for i in range(rows.numel()):
            mask = torch.ones(int(num_classes), dtype=torch.bool)
            mask[ids[i]] = False
            denom = tail_prior[mask].sum().clamp_min(1e-12)
            out[i, mask] = tail[i] * tail_prior[mask] / denom
        return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


def dense_to_stt_cache(probs_or_logits: torch.Tensor, *, mode: str, tail_prior: torch.Tensor | None = None) -> STTTeacherCache:
    probs = _as_probs(probs_or_logits)
    if str(mode) == "dense_fp16":
        return STTTeacherCache(mode=str(mode), dense_probs=probs.to(torch.float16))
    k = min(_k_from_mode(mode), probs.shape[1])
    vals, ids = torch.topk(probs, k=k, dim=1)
    stored_vals = vals.to(torch.float16)
    tail = (1.0 - stored_vals.float().sum(dim=1)).clamp_min(0.0).to(torch.float16)
    prior = tail_prior.float() if tail_prior is not None else probs.mean(dim=0)
    return STTTeacherCache(
        mode=str(mode),
        topk_class_ids=ids.to(torch.int16 if probs.shape[1] <= 32767 else torch.int32),
        topk_probs=stored_vals,
        tail_mass=tail,
        tail_prior=prior,
    )


def estimate_stt_cache_bytes(*, num_nodes: int, num_classes: int, mode: str) -> dict[str, Any]:
    dense = int(num_nodes) * int(num_classes) * 2
    if str(mode) == "dense_fp16":
        cache = dense
    else:
        k = _k_from_mode(mode)
        cache = int(num_nodes) * (k * 2 + k * 2 + 2)
        if "tail" in str(mode):
            cache += int(num_classes) * 4
    return {
        "teacher_cache_bytes": cache,
        "teacher_dense_cache_bytes_diagnostic": dense,
        "cache_compression_ratio": cache / dense if dense else 0.0,
    }
