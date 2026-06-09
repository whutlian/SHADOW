from __future__ import annotations

import torch

from shadow_hgc.prototype.teacher_demand_herding import select_teacher_demand_herding


def test_teacher_demand_herding_respects_budget_and_teacher_gate():
    result = select_teacher_demand_herding(
        embeddings=torch.arange(24, dtype=torch.float32).view(6, 4),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        train_idx=torch.arange(6),
        total_budget=4,
        teacher_valid=False,
        uncertainty=torch.linspace(0.0, 1.0, 6),
        seed=42,
    )

    assert result.indices.numel() == 4
    assert result.diagnostics["teacher_used_for_herding"] is False
    assert result.diagnostics["herding_boundary_count"] == 0
    assert set(result.labels.tolist()) == {0, 1}
