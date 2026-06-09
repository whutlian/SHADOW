import torch

from shadow_hgc.logits.correct_lite import smooth_logits


def test_logit_correct_streaming_matches_dense_destination_row_smoothing():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 1, 0, 2]])

    streaming = smooth_logits(edge_index=edge_index, logits=logits, num_nodes=3, alpha=0.4, steps=1, chunk_size=2)

    dense = torch.zeros(3, 3)
    for src, dst in edge_index.t():
        dense[dst, src] += 1.0
    dense = dense / dense.sum(dim=1, keepdim=True).clamp_min(1.0)
    expected = 0.6 * logits + 0.4 * dense @ logits

    assert torch.allclose(streaming.logits, expected, atol=1e-6)
    assert streaming.diagnostics["normalization"] == "destination_row"
