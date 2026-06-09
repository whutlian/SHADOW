from __future__ import annotations

import torch


def build_train_only_teacher_targets(labels: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    """Return labels with non-train target rows masked out."""

    labels = labels.to(torch.long)
    train_idx = train_idx.to(dtype=torch.long, device=labels.device)
    out = torch.full_like(labels, -1)
    out[train_idx] = labels[train_idx]
    return out


def teacher_label_usage_metadata(*, teacher_type: str, train_idx: torch.Tensor, num_targets: int) -> dict:
    return {
        "type": teacher_type,
        "uses_train_labels_only": True,
        "num_train_label_rows": int(train_idx.numel()),
        "num_target_rows": int(num_targets),
    }
