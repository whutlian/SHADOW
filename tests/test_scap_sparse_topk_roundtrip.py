from __future__ import annotations

import torch

from shadow_hgc.features.scap import dense_from_sparse_topk, sparse_topk_from_dense


def test_scap_sparse_topk_roundtrip_reconstructs_top_values():
    dense = torch.tensor([[0.1, 0.5, 0.2, 0.4], [0.0, -0.2, 0.3, 0.1]], dtype=torch.float32)

    sparse = sparse_topk_from_dense(dense, topk=2)
    restored = dense_from_sparse_topk(sparse, num_classes=4)

    expected = torch.tensor([[0.0, 0.5, 0.0, 0.4], [0.0, 0.0, 0.3, 0.1]], dtype=torch.float32)
    assert torch.allclose(restored, expected)
    assert sparse.metadata["dense_or_sparse"] == "sparse_topk"
    assert sparse.metadata["topk"] == 2
