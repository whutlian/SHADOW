from __future__ import annotations

import torch

from shadow_hgc.sft.simsft_soft import build_simsft_soft_table, simsft_promotion_status


def test_t31_simsft_centroid_rows_use_teacher_soft_averages() -> None:
    features = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.2, 0.8]])
    result = build_simsft_soft_table(features=features, teacher_probs=probs, num_rows=2, method="simsft_soft_centroids", seed=1)
    assert result.z_syn.shape == (2, 2)
    assert torch.allclose(result.y_syn_soft.sum(dim=1), torch.ones(2))
    assert result.diagnostics["uses_full_covariance"] is False
    assert result.diagnostics["uses_exact_pairwise"] is False


def test_t31_simsft_residual_rows_are_bounded() -> None:
    features = torch.tensor([[1.0, 0.0], [5.0, 0.0], [0.0, 1.0], [0.0, 5.0]])
    probs = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    result = build_simsft_soft_table(features=features, teacher_probs=probs, num_rows=4, method="simsft_soft_centroids_plus_residual_exemplars", seed=2)
    assert result.diagnostics["residual_row_count"] > 0
    assert float(result.z_syn.norm(dim=1).max()) <= result.diagnostics["residual_norm_clip"] + float(features.mean(dim=0).norm()) + 1e-6


def test_t31_simsft_promotion_gate_blocks_low_table_only_rows() -> None:
    status, reason = simsft_promotion_status(ratio=0.001, accuracy=0.90)
    assert status == "not_promoted"
    assert reason == "simsft_table_only_gate_not_met"
