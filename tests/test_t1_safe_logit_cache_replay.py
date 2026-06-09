import torch

from shadow_hgc.logits import LogitCacheMeta, save_logits_cache
from shadow_hgc.logits.replay import replay_logits_cache


def test_t1_safe_logit_cache_replay_recomputes_metrics(tmp_path):
    logits = torch.tensor([[4.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 1])
    meta = LogitCacheMeta(
        dataset="toy",
        variant="safe",
        seed=42,
        num_target_nodes=4,
        num_classes=2,
        target_type="paper",
        split_hash="split",
        feature_hash="feature",
        uses_diffusion=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        uses_source_anchors=False,
        uses_coverage_medoid=False,
        uses_old_kd=False,
        accuracy=0.5,
        macro_f1=None,
        predicted_class_count=None,
        created_at="2026-06-09T00:00:00",
    )
    cache_dir = save_logits_cache(
        tmp_path / "cache",
        train_logits=logits[:2],
        valid_logits=logits[2:3],
        test_logits=logits[2:],
        all_target_logits=logits,
        y_train=labels[:2],
        y_valid=labels[2:3],
        y_test=labels[2:],
        train_idx=torch.tensor([0, 1]),
        valid_idx=torch.tensor([2]),
        test_idx=torch.tensor([2, 3]),
        meta=meta,
        dtype="float32",
    )

    replay = replay_logits_cache(cache_dir, historical_test_acc=0.5, tolerance=1e-6)

    assert replay["cache_status"] == "available_verified"
    assert replay["replay_test_acc"] == 0.5
    assert replay["predicted_class_count"] == 2
