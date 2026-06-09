from __future__ import annotations

import torch

from shadow_hgc.features.scap_sparse import dense_from_scap_topk, scap_topk_from_dense


def test_scap_v2_topk_sparse_roundtrip_keeps_top_values():
    dense = torch.tensor([[0.1, 0.8, -0.2, 0.4], [0.0, -1.0, 2.0, 0.5]])

    sparse = scap_topk_from_dense(dense, top_k=2)
    restored = dense_from_scap_topk(sparse)

    expected = torch.tensor([[0.0, 0.8, 0.0, 0.4], [0.0, -1.0, 2.0, 0.0]])
    assert torch.allclose(restored, expected)
    assert sparse.metadata["dense_or_sparse"] == "sparse_topk"
