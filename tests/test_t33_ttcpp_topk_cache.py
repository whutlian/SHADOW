from __future__ import annotations

import torch

from shadow_hgc.sft.ttcpp_topk_cache import TopKTeacherCache, dense_probs_to_topk_cache, load_teacher_cache_for_selection


def test_t33_dense_to_topk_cache_preserves_residual_mass_and_shapes() -> None:
    probs = torch.tensor(
        [
            [0.50, 0.20, 0.20, 0.10],
            [0.05, 0.80, 0.10, 0.05],
        ],
        dtype=torch.float32,
    )
    cache = dense_probs_to_topk_cache(probs, k=2, include_entropy_margin=True)
    assert cache.topk_class_ids.shape == (2, 2)
    assert cache.topk_probs.shape == (2, 2)
    assert cache.residual_mass.shape == (2,)
    assert cache.topk_class_ids.dtype in {torch.int16, torch.int32, torch.int64}
    assert cache.topk_probs.dtype == torch.float16
    assert torch.allclose(cache.residual_mass.float(), 1.0 - cache.topk_probs.float().sum(dim=1), atol=2e-4)
    assert cache.entropy is not None and cache.entropy.shape == (2,)
    assert cache.margin is not None and cache.margin.shape == (2,)


def test_t33_topk_cache_to_dense_reconstructs_accounted_mass() -> None:
    probs = torch.tensor([[0.6, 0.2, 0.1, 0.1]], dtype=torch.float32)
    cache = dense_probs_to_topk_cache(probs, k=2)
    dense = cache.to_dense(num_classes=4)
    assert torch.allclose(dense[0, cache.topk_class_ids[0].long()].sum(), cache.topk_probs[0].float().sum(), atol=1e-5)
    assert abs(float(cache.residual_mass[0]) - 0.2) < 2e-4


def test_t33_ultra_loader_refuses_dense_cache_in_ultra_mode() -> None:
    dense = torch.ones(4, 3) / 3.0
    try:
        load_teacher_cache_for_selection(dense, mode="dense_fp16", ultra_safe=True)
    except ValueError as exc:
        assert "dense teacher cache is not allowed in ultra_safe mode" in str(exc)
    else:
        raise AssertionError("dense cache should be rejected in ultra mode")


def test_t33_topk_cache_byte_estimate_is_smaller_than_dense() -> None:
    cache = TopKTeacherCache.estimate(num_nodes=100, num_classes=40, k=8, include_entropy_margin=True)
    assert cache.teacher_topk_cache_bytes < cache.teacher_dense_cache_bytes_diagnostic
