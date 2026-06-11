from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_teacher_ensemble import (
    build_ensemble_probabilities,
    calibrate_teacher_temperature,
    compute_teacher_diagnostics,
)


def test_t32_temperature_calibration_uses_candidate_grid() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.5, 1.5], [2.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0])
    result = calibrate_teacher_temperature(logits, labels, valid_idx=torch.tensor([0, 1, 2]), temperatures=[1.0, 2.0, 4.0])
    assert result.temperature in {1.0, 2.0, 4.0}
    assert result.valid_nll >= 0.0
    assert result.uses_valid_labels_as_input is False


def test_t32_ensemble_probabilities_sum_to_one_and_identical_disagreement_zero() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    ens = build_ensemble_probabilities([logits, logits], temperatures=[1.0, 1.0])
    assert torch.allclose(ens.probs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(ens.disagreement < 1e-7)


def test_t32_teacher_diagnostics_are_finite() -> None:
    probs = torch.tensor([[0.9, 0.1], [0.55, 0.45]], dtype=torch.float32)
    disagreement = torch.tensor([0.0, 0.2])
    diag = compute_teacher_diagnostics(probs, disagreement)
    assert diag["teacher_entropy_mean"] > 0.0
    assert diag["teacher_margin_mean"] > 0.0
    assert diag["teacher_disagreement_mean"] == 0.1
    assert diag["predicted_classes"] == 1
