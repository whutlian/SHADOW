from __future__ import annotations

import torch

from shadow_hgc.reddit.computation_tree_coverage import ctc_bucket_selection
from shadow_hgc.reddit.condensed_graph_builder import (
    build_knn_graph,
    build_retained_edge_graph,
    destination_row_normalize_edges,
    ensure_self_loops,
)
from shadow_hgc.reddit.edge_predictor import (
    build_edge_candidate_pairs,
    build_pair_features,
    deterministic_negative_pairs,
    edge_predictor_topk_graph,
)
from shadow_hgc.reddit.graph_student import WeightedGraphStudent


def test_t28_knn_graph_has_self_loops_topk_nonnegative_and_dst_row_sums():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    graph = build_knn_graph(x, topk=1, metric="cosine", symmetrize="union", add_self_loops=True)
    assert torch.all(graph.edge_weight >= 0.0)
    for node in range(x.shape[0]):
        assert bool(((graph.edge_index[0] == node) & (graph.edge_index[1] == node)).any())
        incoming = graph.edge_index[1] == node
        assert torch.allclose(graph.edge_weight[incoming].sum(), torch.tensor(1.0), atol=1e-6)
    non_self = graph.edge_index[0] != graph.edge_index[1]
    counts = torch.bincount(graph.edge_index[0][non_self], minlength=x.shape[0])
    assert int(counts.max().item()) <= 2
    assert graph.metadata["edge_weight_normalization"] == "dst_row"


def test_t28_retained_edge_graph_streams_selected_edges_without_dense_full_graph():
    selected = torch.tensor([10, 20, 30])
    chunks = [
        (torch.tensor([10, 20, 99]), torch.tensor([20, 10, 30])),
        (torch.tensor([30, 20]), torch.tensor([20, 99])),
    ]
    graph = build_retained_edge_graph(selected_node_ids=selected, edge_chunks=chunks, add_self_loops=True)
    assert graph.metadata["full_edge_scans"] == 1
    assert graph.metadata["loads_edge_index"] is False
    assert graph.metadata["uses_full_edge_index_on_gpu"] is False
    assert graph.metadata["retained_original_edges"] == 3
    assert torch.all(graph.edge_weight >= 0.0)


def test_t28_edge_predictor_sampling_and_topk_are_deterministic():
    labels = torch.tensor([0, 0, 1, 1])
    positives = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    neg_a = deterministic_negative_pairs(num_nodes=4, positive_edge_index=positives, labels=labels, num_pairs=4, seed=7)
    neg_b = deterministic_negative_pairs(num_nodes=4, positive_edge_index=positives, labels=labels, num_pairs=4, seed=7)
    assert torch.equal(neg_a, neg_b)
    features = build_pair_features(
        torch.eye(4),
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        labels=labels,
        degree=torch.arange(4, dtype=torch.float32),
        use_same_label=True,
    )
    assert features.shape[0] == 3
    candidates = build_edge_candidate_pairs(torch.eye(4), labels=labels, max_candidates_per_node=2, seed=42)
    graph = edge_predictor_topk_graph(candidates, torch.linspace(0.1, 0.9, candidates.shape[1]), num_nodes=4, topk=1, add_self_loops=True)
    non_self = graph.edge_index[0] != graph.edge_index[1]
    counts = torch.bincount(graph.edge_index[1][non_self], minlength=4)
    assert int(counts.max().item()) <= 1
    assert graph.metadata["uses_edge_predictor"] is True
    assert graph.metadata["edge_predictor_train_pairs"] == int(candidates.shape[1])


def test_t28_ctc_selection_is_deterministic_and_exact_budget():
    signature = torch.arange(60, dtype=torch.float32).view(10, 6)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    degree = torch.tensor([1, 2, 4, 8, 16, 32, 64, 3, 6, 9])
    first = ctc_bucket_selection(signature, labels, total_budget=6, degree=degree, seed=123)
    second = ctc_bucket_selection(signature, labels, total_budget=6, degree=degree, seed=123)
    assert torch.equal(first.selected_pos, second.selected_pos)
    assert first.selected_pos.numel() == 6
    assert first.diagnostics["ctc_num_buckets"] > 0
    assert first.diagnostics["runs_full_all_pair_original_search"] is False


def test_t28_weighted_graph_student_uses_explicit_weights_without_double_norm():
    model = WeightedGraphStudent(input_dim=2, hidden_dim=2, num_classes=2, layers=1, model_type="weighted_gcn", residual=False, norm="none", dropout=0.0)
    with torch.no_grad():
        linear = model.layers[0]
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([0.25, 0.75, 1.0])
    out = model(x, edge_index, edge_weight)
    expected_hidden = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.25, 1.5]])
    assert torch.allclose(out, expected_hidden)
    assert model.uses_library_normalization is False


def test_t28_destination_row_normalize_edges_handles_zero_in_degree():
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    weight = destination_row_normalize_edges(edge_index, torch.ones(2), num_nodes=4)
    assert torch.allclose(weight, torch.tensor([0.5, 0.5]))
    with_self = ensure_self_loops(edge_index, num_nodes=4)
    assert with_self.shape[1] >= edge_index.shape[1] + 4
