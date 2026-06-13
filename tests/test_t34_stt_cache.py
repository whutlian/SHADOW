from __future__ import annotations

import torch

from shadow_hgc.sft.stt_cache import dense_to_stt_cache, estimate_stt_cache_bytes


def test_t34_topk_tail_reconstructs_selected_rows_to_probability_simplex() -> None:
    probs = torch.tensor([[0.6, 0.2, 0.1, 0.1], [0.05, 0.8, 0.1, 0.05]], dtype=torch.float32)
    cache = dense_to_stt_cache(probs, mode="topk2_tail", tail_prior=probs.mean(dim=0))
    dense = cache.reconstruct_rows(torch.tensor([0, 1]), num_classes=4)
    assert dense.shape == (2, 4)
    assert torch.allclose(dense.sum(dim=1), torch.ones(2), atol=1e-4)
    assert torch.all(cache.tail_mass.float() >= 0)
    assert torch.all(cache.tail_mass.float() <= 1)


def test_t34_dense_fp16_reconstructs_selected_rows() -> None:
    probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
    cache = dense_to_stt_cache(probs, mode="dense_fp16")
    dense = cache.reconstruct_rows(torch.tensor([1]), num_classes=2)
    assert torch.allclose(dense, probs[1:2], atol=3e-4)


def test_t34_cache_estimate_topk_tail_smaller_than_dense() -> None:
    estimate = estimate_stt_cache_bytes(num_nodes=1000, num_classes=100, mode="topk8_tail")
    assert estimate["teacher_cache_bytes"] < estimate["teacher_dense_cache_bytes_diagnostic"]
    assert estimate["cache_compression_ratio"] < 0.5
