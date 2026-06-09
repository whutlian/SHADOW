import numpy as np
import torch

from shadow_hgc.logits import LogitCacheMeta, load_logits_cache, save_logits_cache


def test_logit_cache_roundtrip_preserves_arrays_and_metadata(tmp_path):
    meta = LogitCacheMeta(
        dataset="toy",
        variant="safe_base",
        seed=42,
        num_target_nodes=4,
        num_classes=3,
        target_type="paper",
        split_hash="s",
        feature_hash="f",
        uses_diffusion=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        uses_source_anchors=False,
        uses_coverage_medoid=False,
        uses_old_kd=False,
        accuracy=0.5,
        macro_f1=0.4,
        predicted_class_count=3,
        created_at="2026-06-09T00:00:00",
    )

    path = save_logits_cache(
        tmp_path,
        train_logits=torch.randn(2, 3),
        valid_logits=np.ones((1, 3), dtype=np.float32),
        test_logits=torch.zeros(1, 3),
        all_target_logits=torch.arange(12, dtype=torch.float32).view(4, 3),
        y_train=torch.tensor([0, 1]),
        y_valid=torch.tensor([2]),
        y_test=torch.tensor([1]),
        train_idx=torch.tensor([0, 1]),
        valid_idx=torch.tensor([2]),
        test_idx=torch.tensor([3]),
        meta=meta,
        dtype="float16",
    )

    loaded = load_logits_cache(path)

    assert loaded.meta.dataset == "toy"
    assert loaded.all_target_logits.shape == (4, 3)
    assert loaded.train_logits.shape == (2, 3)
    assert loaded.y_train.tolist() == [0, 1]
    assert loaded.train_idx.tolist() == [0, 1]
