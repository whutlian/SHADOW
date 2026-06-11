from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


class CacheEstimate(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_probs(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().float()
    row_sum = x.sum(dim=1, keepdim=True)
    if bool(torch.all(x >= 0).item()) and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)):
        return x / row_sum.clamp_min(1e-12)
    return torch.softmax(x, dim=1)


@dataclass(frozen=True)
class TopKTeacherCache:
    topk_class_ids: torch.Tensor
    topk_probs: torch.Tensor
    residual_mass: torch.Tensor
    entropy: torch.Tensor | None = None
    margin: torch.Tensor | None = None

    def to_dense(self, *, num_classes: int) -> torch.Tensor:
        out = torch.zeros((int(self.topk_class_ids.shape[0]), int(num_classes)), dtype=torch.float32)
        rows = torch.arange(out.shape[0]).view(-1, 1).expand_as(self.topk_class_ids)
        out[rows, self.topk_class_ids.long()] = self.topk_probs.float()
        return out

    @staticmethod
    def estimate(*, num_nodes: int, num_classes: int, k: int, include_entropy_margin: bool = False) -> CacheEstimate:
        id_bytes = int(num_nodes) * int(k) * 2
        prob_bytes = int(num_nodes) * int(k) * 2
        residual_bytes = int(num_nodes) * 2
        aux_bytes = int(num_nodes) * 2 * (2 if include_entropy_margin else 0)
        dense_bytes = int(num_nodes) * int(num_classes) * 2
        return CacheEstimate({
            "teacher_topk_cache_bytes": id_bytes + prob_bytes + residual_bytes + aux_bytes,
            "teacher_dense_cache_bytes_diagnostic": dense_bytes,
        })


def dense_probs_to_topk_cache(probs_or_logits: torch.Tensor, *, k: int, include_entropy_margin: bool = False) -> TopKTeacherCache:
    probs = _as_probs(probs_or_logits)
    kk = min(int(k), int(probs.shape[1]))
    values, ids = torch.topk(probs, k=kk, dim=1)
    stored_values = values.to(torch.float16)
    residual = (1.0 - stored_values.float().sum(dim=1)).clamp_min(0.0)
    entropy = None
    margin = None
    if include_entropy_margin:
        entropy = (-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)).to(torch.float16)
        top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
        margin = (top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)).to(torch.float16)
    return TopKTeacherCache(
        topk_class_ids=ids.to(torch.int16 if probs.shape[1] <= 32767 else torch.int32),
        topk_probs=stored_values,
        residual_mass=residual.to(torch.float16),
        entropy=entropy,
        margin=margin,
    )


def load_teacher_cache_for_selection(cache: torch.Tensor | TopKTeacherCache, *, mode: str, ultra_safe: bool = False) -> torch.Tensor | TopKTeacherCache:
    if bool(ultra_safe) and str(mode) == "dense_fp16":
        raise ValueError("dense teacher cache is not allowed in ultra_safe mode")
    if isinstance(cache, TopKTeacherCache):
        return cache
    if str(mode).startswith("topk"):
        digits = "".join(ch for ch in str(mode) if ch.isdigit())
        k = int(digits) if digits else 8
        return dense_probs_to_topk_cache(cache, k=k, include_entropy_margin="entropy_margin" in str(mode))
    return _as_probs(cache).to(torch.float16 if str(mode) == "dense_fp16" else torch.float32)


def teacher_cache_hash(values: torch.Tensor | TopKTeacherCache) -> str:
    digest = hashlib.sha256()
    if isinstance(values, TopKTeacherCache):
        for tensor in (values.topk_class_ids, values.topk_probs, values.residual_mass):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    else:
        digest.update(values.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]
