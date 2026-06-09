from __future__ import annotations

import torch

from shadow_hgc.teacher.cache import TeacherCache, build_teacher_cache


def test_teacher_cache_shapes_and_roundtrip(tmp_path):
    cache = build_teacher_cache(
        logits=torch.randn(4, 3),
        train_idx=torch.tensor([0, 2, 3, 5]),
        embeddings=torch.randn(4, 8),
        teacher_type="sehgnn_lite",
        metadata={"train_acc": 0.75},
    )
    path = tmp_path / "teacher.pt"
    cache.save(path)
    loaded = TeacherCache.load(path)

    assert loaded.logits.shape == (4, 3)
    assert loaded.embeddings is not None
    assert loaded.embeddings.shape == (4, 8)
    assert loaded.train_idx.tolist() == [0, 2, 3, 5]
    assert loaded.metadata["teacher_type"] == "sehgnn_lite"
