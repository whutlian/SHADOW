from __future__ import annotations

import torch

from shadow_hgc.sft.codebook_assignment import build_codebook_assignment
from shadow_hgc.sft.quotient_operator import build_quotient_operator, quotient_to_dense


def test_t30_codebook_assignment_covers_every_original_node() -> None:
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
            [9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 1])
    train_idx = torch.tensor([0, 2, 3])
    result = build_codebook_assignment(
        features=features,
        labels=labels,
        train_idx=train_idx,
        num_codewords=4,
        num_classes=2,
        mode="qoc_class_conditional_online_kmeans",
        seed=7,
    )
    assert result.assignments.shape == (5,)
    assert int((result.assignments < 0).sum().item()) == 0
    assert int(result.diagnostics["num_assigned_nodes"]) == 5
    assert int(result.diagnostics["num_unassigned_nodes"]) == 0
    assert result.codebook_train_label_mass.shape == (4, 2)


def test_t30_quotient_operator_orientation_destination_rows_and_repair() -> None:
    # Original edges are source -> destination. Nodes 0/1 map to code 0, nodes 2/3 map to code 1.
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long)
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    result = build_quotient_operator(
        edge_index=edge_index,
        assignments=assignments,
        num_codewords=2,
        topk=2,
        mode="original_dest_normalized",
    )
    dense = quotient_to_dense(result.edge_index, result.edge_weight, num_codewords=2)
    # Row 0 has no incoming original support and is repaired with a self-loop.
    assert torch.allclose(dense[0], torch.tensor([1.0, 0.0]))
    # Row 1 receives half from code 0 and half from code 1 after destination normalization.
    assert torch.allclose(dense[1], torch.tensor([0.5, 0.5]))
    assert result.diagnostics["operator_zero_rows"] == 1
    assert result.diagnostics["operator_repaired_rows"] == 1
    assert result.diagnostics["operator_row_sum_error"] <= 1e-6


def test_t30_quotient_operator_topk_prunes_then_renormalizes_rows() -> None:
    edge_index = torch.tensor([[0, 1, 2, 0], [3, 3, 3, 2]], dtype=torch.long)
    assignments = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    result = build_quotient_operator(
        edge_index=edge_index,
        assignments=assignments,
        num_codewords=4,
        topk=1,
        mode="code_row_normalized_fallback",
    )
    dense = quotient_to_dense(result.edge_index, result.edge_weight, num_codewords=4)
    assert torch.allclose(dense.sum(dim=1), torch.ones(4))
    assert int((dense > 0).sum(dim=1).max().item()) == 1
    assert result.diagnostics["operator_edges_before_topk"] >= result.diagnostics["operator_edges_after_topk"]
    assert result.diagnostics["uses_dense_adjacency"] is False
