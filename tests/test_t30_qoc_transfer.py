from __future__ import annotations

import torch

from shadow_hgc.sft.qoc_condense import build_qoc_table
from shadow_hgc.sft.qoc_pltc import aggregate_teacher_soft_labels, qoc_pltc_split
from shadow_hgc.sft.qoc_transfer_eval import train_qoc_table_head


def test_t30_qoc_table_builds_operator_features_without_dense_adjacency() -> None:
    z0 = torch.eye(3, dtype=torch.float32)
    y0 = torch.eye(3, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.ones(3, dtype=torch.float32)
    table, diag = build_qoc_table(z0=z0, y0=y0, edge_index=edge_index, edge_weight=edge_weight)
    assert table.shape == (3, 18)
    assert diag["uses_dense_adjacency"] is False
    assert diag["table_blocks"] == "Z0,PZ0,P2Z0,Y0,PY0,P2Y0"


def test_t30_qoc_table_head_reports_real_transfer_eval_metrics() -> None:
    input_syn = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.8, 0.1], [0.2, 1.7]])
    labels_syn = torch.tensor([0, 1, 0, 1])
    weights = torch.ones(4)
    input_real = torch.tensor([[2.2, 0.1], [0.1, 2.1], [1.7, 0.0], [0.0, 1.8]])
    labels_real = torch.tensor([0, 1, 0, 1])
    result = train_qoc_table_head(
        input_syn=input_syn,
        labels_syn=labels_syn,
        code_weights=weights,
        input_real=input_real,
        labels_real=labels_real,
        num_classes=2,
        hidden_dim=8,
        epochs=80,
        seed=3,
    )
    assert result.metrics["transfer_eval_type"] == "real_transfer_eval"
    assert result.metrics["accuracy"] >= 0.75
    assert result.metrics["macro_f1"] >= 0.75
    assert result.metrics["uses_full_edge_index_on_gpu"] is False


def test_t30_pltc_soft_labels_are_sota_only_and_train_free() -> None:
    assignments = torch.tensor([0, 0, 1, 1])
    teacher_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    soft, diag = aggregate_teacher_soft_labels(assignments, teacher_probs, num_codewords=2)
    split = qoc_pltc_split(num_codewords=20)
    assert torch.allclose(soft.sum(dim=1), torch.ones(2))
    assert diag["uses_teacher_logits"] is True
    assert diag["uses_valid_labels_as_input"] is False
    assert diag["uses_test_labels_as_input"] is False
    assert sum(split.values()) == 20
    assert split["confident_pseudo_class_codewords"] == 10
