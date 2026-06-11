from __future__ import annotations

import torch

from scripts.run_t29_reddit_operator_match import build_omcp_rows
from shadow_hgc.reddit.operator_students import OperatorSFTTableHead, WeightedOperatorStudent
from shadow_hgc.sft.operator_match import (
    apply_sparse_operator,
    build_knn_candidate_edges,
    edge_weights_by_dst_softmax,
    fit_operator_match,
    project_topk_by_dst,
)


def test_t29_operator_row_stochastic_and_nonnegative():
    edge_index = torch.tensor([[0, 1, 2, 0, 2], [0, 0, 1, 2, 2]], dtype=torch.long)
    logits = torch.tensor([0.1, 0.2, -0.3, 1.0, 0.5])
    weight = edge_weights_by_dst_softmax(edge_index, logits, num_nodes=3)
    assert torch.all(weight >= 0)
    for dst in range(3):
        mask = edge_index[1] == dst
        assert torch.allclose(weight[mask].sum(), torch.tensor(1.0), atol=1e-6)


def test_t29_operator_topk_projection_and_direction_source_to_destination():
    edge_index = torch.tensor([[0, 1, 2, 0, 2], [0, 0, 1, 2, 2]], dtype=torch.long)
    weight = torch.tensor([0.2, 0.8, 1.0, 0.4, 0.6])
    projected_index, projected_weight = project_topk_by_dst(edge_index, weight, num_nodes=3, topk=1)
    assert projected_index.shape[1] == 3
    assert (projected_index[0] == torch.tensor([1, 2, 2])).all()
    assert (projected_index[1] == torch.tensor([0, 1, 2])).all()
    for dst in range(3):
        assert torch.allclose(projected_weight[projected_index[1] == dst].sum(), torch.tensor(1.0), atol=1e-6)


def test_t29_operator_matches_simple_one_hop_block():
    x0 = torch.eye(3)
    edge_index = torch.tensor([[1, 2, 0], [0, 1, 2]], dtype=torch.long)
    target = apply_sparse_operator(x0, edge_index, torch.ones(3))
    result = fit_operator_match(
        x0=x0,
        x1_target=target,
        candidate_edge_index=edge_index,
        topk=1,
        steps=120,
        lr=0.05,
        seed=42,
    )
    pred = apply_sparse_operator(x0, result.edge_index, result.edge_weight)
    assert torch.mean((pred - target) ** 2).item() < 1e-4
    assert result.diagnostics["operator_negative_weight_count"] == 0
    assert result.diagnostics["operator_row_sum_error"] < 1e-4


def test_t29_operator_no_dense_adjacency_for_promoted_rows():
    x = torch.randn(8, 4)
    candidates = build_knn_candidate_edges(x, candidate_topk=3)
    assert candidates.metadata["uses_dense_adjacency"] is False
    assert candidates.metadata["uses_exact_pairwise"] is False


def test_t29_operator_student_no_double_normalization():
    student = WeightedOperatorStudent(input_dim=2, hidden_dim=2, num_classes=2, model_type="weighted_sgc", layers=1, dropout=0.0, norm="none")
    assert student.uses_library_normalization is False
    table = OperatorSFTTableHead(input_dim=4, num_classes=2)
    out = table(torch.ones(3, 4))
    assert out.shape == (3, 2)


def test_t29_omcp_runner_budget_scales_with_ratio():
    args = type(
        "Args",
        (),
        {
            "ratios": [0.001, 0.005],
            "prototype_inits": ["current_sft_signature_random"],
            "operator_topks": [4],
            "students": ["operator_sft_table_head"],
            "seed": 42,
            "smoke": True,
        },
    )()
    rows = build_omcp_rows(args)
    counts = {row["requested_full_node_ratio"]: row["actual_condensed_nodes"] for row in rows}
    assert counts[0.001] == 233
    assert counts[0.005] == 1165
