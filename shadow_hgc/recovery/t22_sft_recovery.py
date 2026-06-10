from __future__ import annotations

from typing import Any

import torch


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor) -> float:
    rows = rows.to(torch.long)
    if rows.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1).to(torch.long)
    return float((pred[rows] == labels[rows].to(torch.long)).to(torch.float32).mean().item())


def identity_replay_gap(*, full_logits: torch.Tensor, replay_logits: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor) -> dict[str, Any]:
    rows = rows.to(torch.long)
    full_acc = _accuracy(full_logits, labels, rows)
    replay_acc = _accuracy(replay_logits, labels, rows)
    full_pred = full_logits.argmax(dim=1).to(torch.long)[rows]
    replay_pred = replay_logits.argmax(dim=1).to(torch.long)[rows]
    mismatch = float((full_pred != replay_pred).to(torch.float32).mean().item()) if rows.numel() else 0.0
    return {
        "full_accuracy": full_acc,
        "replay_accuracy": replay_acc,
        "accuracy_gap": abs(full_acc - replay_acc),
        "prediction_mismatch_rate": mismatch,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
    }
