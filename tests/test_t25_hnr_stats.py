import torch

from shadow_hgc.data.edge_stream import ArrayEdgeStream
from shadow_hgc.sft.hnr import compute_streaming_hnr_stats


def _stream(src, dst):
    return lambda: ArrayEdgeStream(
        src=torch.tensor(src, dtype=torch.long).numpy(),
        dst=torch.tensor(dst, dtype=torch.long).numpy(),
        chunk_size=2,
    )


def test_t25_hnr_directed_stats_match_toy_graph():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    train_rows = torch.tensor([0, 1, 2, 3])
    target_rows = torch.tensor([0, 1, 2, 3])
    stats = compute_streaming_hnr_stats(
        edge_stream_factory=_stream([1, 2, 4, 0, 3], [0, 0, 0, 2, 2]),
        num_nodes=6,
        labels=labels,
        train_rows=train_rows,
        target_rows=target_rows,
        num_classes=3,
    )

    assert stats.degree.tolist() == [3, 0, 2, 0]
    assert stats.labeled_support.tolist() == [2, 0, 2, 0]
    assert stats.same_label_support.tolist() == [1, 0, 1, 0]
    assert torch.allclose(stats.homophily, torch.tensor([0.5, 0.0, 0.5, 0.0]))
    assert torch.allclose(stats.label_max_affinity, torch.tensor([0.5, 0.0, 0.5, 0.0]))
    assert set(stats.stratum) <= {"H+", "H0", "H-"}


def test_t25_hnr_is_direction_sensitive():
    labels = torch.tensor([0, 0, 1, 1])
    train_rows = torch.arange(4)
    forward = compute_streaming_hnr_stats(
        edge_stream_factory=_stream([0, 1, 2], [3, 3, 3]),
        num_nodes=4,
        labels=labels,
        train_rows=train_rows,
        target_rows=train_rows,
        num_classes=2,
    )
    reversed_stats = compute_streaming_hnr_stats(
        edge_stream_factory=_stream([3, 3, 3], [0, 1, 2]),
        num_nodes=4,
        labels=labels,
        train_rows=train_rows,
        target_rows=train_rows,
        num_classes=2,
    )

    assert forward.degree.tolist() != reversed_stats.degree.tolist()
    assert forward.degree[3].item() == 3
    assert reversed_stats.degree[3].item() == 0


def test_t25_hnr_uses_train_labels_only():
    train_rows = torch.tensor([0, 1])
    target_rows = torch.tensor([0, 1, 2, 3])
    labels_a = torch.tensor([0, 1, 0, 1])
    labels_b = torch.tensor([0, 1, 1, 0])
    kwargs = dict(
        edge_stream_factory=_stream([2, 3, 1, 0], [0, 0, 2, 3]),
        num_nodes=4,
        train_rows=train_rows,
        target_rows=target_rows,
        num_classes=2,
    )

    stats_a = compute_streaming_hnr_stats(labels=labels_a, **kwargs)
    stats_b = compute_streaming_hnr_stats(labels=labels_b, **kwargs)

    assert torch.equal(stats_a.labeled_support, stats_b.labeled_support)
    assert torch.equal(stats_a.same_label_support, stats_b.same_label_support)
    assert torch.allclose(stats_a.homophily, stats_b.homophily)
