from __future__ import annotations

import torch

from shadow_hgc.sft.stt_selection_streaming import select_stt_streaming


def test_t34_streaming_selection_respects_budget_and_no_valid_test_leakage() -> None:
    features = torch.eye(8, dtype=torch.float32)
    probs = torch.tensor(
        [[0.9, 0.1], [0.85, 0.15], [0.1, 0.9], [0.15, 0.85], [0.55, 0.45], [0.45, 0.55], [0.7, 0.3], [0.3, 0.7]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    train = torch.tensor([0, 2, 4])
    valid = torch.tensor([1, 3])
    test = torch.tensor([5, 6, 7])
    first = select_stt_streaming(features=features, teacher_probs=probs, labels=labels, train_idx=train, valid_idx=valid, test_idx=test, num_rows=4, ratio=0.001, seed=11)
    changed = labels.clone()
    changed[valid] = 1 - changed[valid]
    changed[test] = 1 - changed[test]
    second = select_stt_streaming(features=features, teacher_probs=probs, labels=changed, train_idx=train, valid_idx=valid, test_idx=test, num_rows=4, ratio=0.001, seed=11)
    assert first.z_syn.shape[0] == 4
    assert first.source_node_ids.tolist() == second.source_node_ids.tolist()
    assert torch.allclose(first.y_syn_soft, second.y_syn_soft)
    assert first.diagnostics["uses_global_sort"] is False
    assert first.diagnostics["candidate_nodes_mode"] == "all"


def test_t34_streaming_selection_logs_bucket_fractions() -> None:
    features = torch.eye(6, dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.45, 0.55]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    table = select_stt_streaming(features=features, teacher_probs=probs, labels=labels, train_idx=torch.tensor([0, 2]), num_rows=3, ratio=0.005, seed=2)
    assert 0.0 <= table.diagnostics["core_frac_actual"] <= 1.0
    assert "boundary_frac_actual" in table.diagnostics
    assert table.diagnostics["total_condensed_nodes"] == 3
