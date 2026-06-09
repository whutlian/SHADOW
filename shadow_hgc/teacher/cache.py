from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TeacherCache:
    logits: torch.Tensor
    train_idx: torch.Tensor
    embeddings: torch.Tensor | None
    metadata: dict

    def save(self, path: str | Path) -> None:
        payload = {
            "logits": self.logits.detach().cpu(),
            "train_idx": self.train_idx.detach().cpu(),
            "embeddings": None if self.embeddings is None else self.embeddings.detach().cpu(),
            "metadata": dict(self.metadata),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "TeacherCache":
        payload = torch.load(Path(path), map_location="cpu")
        return cls(
            logits=payload["logits"],
            train_idx=payload["train_idx"].to(torch.long),
            embeddings=payload.get("embeddings"),
            metadata=dict(payload.get("metadata", {})),
        )


def build_teacher_cache(
    *,
    logits: torch.Tensor,
    train_idx: torch.Tensor,
    embeddings: torch.Tensor | None = None,
    teacher_type: str,
    metadata: dict | None = None,
) -> TeacherCache:
    if logits.ndim != 2:
        raise ValueError("teacher logits must be rank-2")
    train_idx = train_idx.to(torch.long)
    if int(logits.shape[0]) != int(train_idx.numel()):
        raise ValueError("teacher logits rows must match train_idx length")
    if embeddings is not None and int(embeddings.shape[0]) != int(train_idx.numel()):
        raise ValueError("teacher embeddings rows must match train_idx length")
    meta = dict(metadata or {})
    meta["teacher_type"] = teacher_type
    meta["cache_rows"] = int(train_idx.numel())
    meta["num_classes"] = int(logits.shape[1])
    return TeacherCache(
        logits=logits.to(torch.float32).detach().cpu(),
        train_idx=train_idx.detach().cpu(),
        embeddings=None if embeddings is None else embeddings.to(torch.float32).detach().cpu(),
        metadata=meta,
    )
