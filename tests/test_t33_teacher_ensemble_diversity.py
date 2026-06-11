from __future__ import annotations

import torch

from shadow_hgc.sft.t33_teacher_ensemble import cache_tensor_hash, teacher_ensemble_diversity


def test_t33_duplicate_teacher_cache_hashes_are_detected() -> None:
    probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
    diag = teacher_ensemble_diversity([probs, probs.clone()], teacher_ids=["a", "b"])
    assert diag["teacher_ensemble_size"] == 2
    assert diag["teacher_cache_duplicate_detected"] is True
    assert diag["teacher_disagreement_mean"] <= 1e-6
    assert diag["ensemble_failure_reason"] == "ensemble_not_diverse"
    assert len(diag["teacher_cache_hashes"].split(";")) == 2


def test_t33_nonidentical_teacher_pairwise_kl_is_positive() -> None:
    first = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
    second = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
    diag = teacher_ensemble_diversity([first, second], teacher_ids=["a", "b"])
    assert diag["teacher_cache_duplicate_detected"] is False
    assert diag["teacher_pairwise_kl_mean"] > 0.0
    assert diag["teacher_pairwise_kl_max"] >= diag["teacher_pairwise_kl_min"]
    assert cache_tensor_hash(first) != cache_tensor_hash(second)
