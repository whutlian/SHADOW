from __future__ import annotations

import torch

from shadow_hgc.features.scap import target_target_scap_dense, target_target_scap_streaming


def test_scap_target_target_dense_matches_streaming_chunks():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [2, 2, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([0, 1, -1, 2, -1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, True, False])
    target_rows = torch.arange(5)

    dense = target_target_scap_dense(
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        num_nodes=5,
        num_classes=3,
        target_rows=target_rows,
    )
    streaming = target_target_scap_streaming(
        edge_chunks=[edge_index[:, :2], edge_index[:, 2:4], edge_index[:, 4:]],
        labels=labels,
        train_mask=train_mask,
        num_nodes=5,
        num_classes=3,
        target_rows=target_rows,
    )

    assert torch.allclose(dense, streaming, atol=1e-6)
    assert dense[2].tolist() == [0.5, 0.5, 0.0]
