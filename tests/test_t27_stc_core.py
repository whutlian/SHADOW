import json

import numpy as np
import pytest
import torch

from shadow_hgc.sft.stc import (
    BlockSpec,
    apply_tanh_bounded_delta,
    class_histogram_json,
    count_nonzero_histogram,
    delta_bound_ratios,
    products_balanced_gate,
)
from shadow_hgc.sft.stc_contract import (
    T25_T26_DIAGNOSTIC_METHODS,
    T27_REQUIRED_FIELDS,
    apply_t27_promotion_guard,
    make_t27_row,
    validate_t27_promoted_row,
)
from shadow_hgc.sft.stc_init import class_balanced_random_init
from shadow_hgc.sft.stc_trainer import optimize_gradient_matching, optimize_trainable_delta
from shadow_hgc.sft.timeaware_arxiv import apply_arxiv_teacher_gate, temporal_labelreuse_decay


def _toy_tables():
    generator = torch.Generator().manual_seed(11)
    centers = torch.tensor(
        [
            [-2.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    z_real = torch.cat([center + 0.08 * torch.randn(24, 4, generator=generator) for center in centers], dim=0)
    y_real = torch.cat([torch.full((24,), cls, dtype=torch.long) for cls in range(3)], dim=0)
    z_init = torch.stack([z_real[y_real == cls].mean(dim=0) * 0.45 for cls in range(3)], dim=0)
    y_syn = torch.arange(3, dtype=torch.long)
    return z_real, y_real, z_init, y_syn


def test_t27_row_has_structure_free_ratio_and_forbidden_guards():
    row = make_t27_row(
        dataset="toy",
        method="sft_stc_trainable_delta",
        seed=42,
        requested_full_node_ratio=0.005,
        original_num_nodes=1000,
        num_train_nodes=100,
        num_classes=3,
        syn_rows=5,
        syn_feature_dim=8,
        accuracy=0.9,
        macro_f1=0.8,
        predicted_classes=3,
        promotion_status="promoted",
    )

    assert row["ratio_mode"] == "full_node"
    assert row["actual_full_node_ratio"] == pytest.approx(0.005)
    assert row["shadow_nodes"] == 0
    assert row["condensed_edges"] == 0
    assert row["target_prototypes"] == 5
    assert row["promotion_allowed"] is True
    assert validate_t27_promoted_row(row)["valid"] is True

    row["uses_teacher_logits"] = True
    guarded = apply_t27_promotion_guard(row, dataset_gate_passed=True)
    assert guarded["promotion_allowed"] is False
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_logits" in guarded["failure_reason"]


def test_t27_required_fields_cover_outputs_and_demoted_methods():
    for field in [
        "dataset",
        "method",
        "ratio_mode",
        "actual_full_node_ratio",
        "syn_rows",
        "uses_teacher_logits",
        "promotion_allowed",
        "failure_reason",
    ]:
        assert field in T27_REQUIRED_FIELDS
    assert "sft_hnr_fdm_hybrid" in T25_T26_DIAGNOSTIC_METHODS


def test_tanh_bounded_delta_respects_block_rho():
    init = torch.tensor([[1.0, 2.0, 10.0, 0.5], [2.0, 1.0, 8.0, 1.5]], dtype=torch.float32)
    raw = torch.full_like(init, 100.0)
    blocks = [BlockSpec("a", 0, 2), BlockSpec("b", 2, 4)]

    z_syn, delta = apply_tanh_bounded_delta(init, raw, blocks, rho=0.05)
    ratios = delta_bound_ratios(init, delta, blocks)

    assert max(ratios.values()) <= 0.050001
    assert z_syn.shape == init.shape
    assert torch.allclose(z_syn, init + delta)


def test_t27_trainable_delta_toy_improves_real_train_loss():
    z_real, y_real, z_init, y_syn = _toy_tables()
    blocks = [BlockSpec("all", 0, z_real.shape[1])]

    result = optimize_trainable_delta(
        z_init,
        y_syn,
        z_real,
        y_real,
        blocks,
        num_classes=3,
        rho=0.20,
        outer_steps=30,
        lr=0.05,
        seed=7,
    )

    assert result.final_real_loss < result.initial_real_loss
    assert result.z_syn.shape == z_init.shape
    assert result.delta_bound_ratios["all"] <= 0.200001


def test_t27_gradient_matching_updates_synthetic_features():
    z_real, y_real, z_init, y_syn = _toy_tables()
    blocks = [BlockSpec("all", 0, z_real.shape[1])]

    result = optimize_gradient_matching(
        z_init,
        y_syn,
        z_real,
        y_real,
        blocks,
        num_classes=3,
        rho=0.10,
        outer_steps=8,
        real_batch_size=24,
        lr=0.03,
        seed=13,
    )

    assert not torch.allclose(result.z_syn, z_init)
    assert result.delta_bound_ratios["all"] <= 0.100001
    assert result.used_valid_labels is False
    assert result.used_test_labels is False


def test_stc_initialization_ignores_valid_and_test_labels():
    table = np.arange(60, dtype=np.float32).reshape(10, 6)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    train_idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    out1 = class_balanced_random_init(table, labels, train_idx, m=6, seed=3)
    labels[6:] = np.array([2, 2, 2, 2], dtype=np.int64)
    out2 = class_balanced_random_init(table, labels, train_idx, m=6, seed=3)

    assert np.array_equal(out1.row_ids, out2.row_ids)
    assert out1.z_init.shape == (6, 6)
    assert set(out1.y_init.tolist()) == {0, 1, 2}


def test_products_histogram_and_balanced_gate():
    histogram = class_histogram_json(np.array([0, 0, 2, 5, 5, 5]), num_classes=7)
    parsed = json.loads(histogram)

    assert parsed["0"] == 2
    assert parsed["1"] == 0
    assert count_nonzero_histogram(histogram) == 3
    assert products_balanced_gate(
        predicted_class_histogram_json=histogram,
        macro_f1=0.41,
        accuracy=0.74,
        min_predicted_classes=4,
    ) is False


def test_arxiv_condensation_cannot_promote_without_a1():
    row = make_t27_row(
        dataset="ogbn-arxiv",
        method="arxiv_random_gm",
        seed=42,
        requested_full_node_ratio=0.005,
        original_num_nodes=169343,
        num_train_nodes=90941,
        num_classes=40,
        syn_rows=847,
        syn_feature_dim=256,
        accuracy=0.712,
        macro_f1=0.50,
        predicted_classes=39,
        promotion_status="promoted",
        extra={"valid_acc": 0.713, "A1_passed": False},
    )

    guarded = apply_arxiv_teacher_gate(row)
    assert guarded["A1_passed"] is False
    assert guarded["promotion_allowed"] is False
    assert guarded["promotion_status"] == "blocked_teacher_gate"


def test_temporal_labelreuse_decay_uses_train_labels_only():
    edge_index = np.array([[0, 1, 2, 3, 4], [2, 2, 3, 4, 5]], dtype=np.int64)
    years = np.array([2016, 2017, 2018, 2018, 2019, 2020], dtype=np.int64)
    labels_a = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    train_mask = np.array([True, True, False, True, False, False])

    out1 = temporal_labelreuse_decay(edge_index, years, labels_a, train_mask, num_classes=3, gamma=0.1)
    labels_b = labels_a.copy()
    labels_b[~train_mask] = (labels_b[~train_mask] + 1) % 3
    out2 = temporal_labelreuse_decay(edge_index, years, labels_b, train_mask, num_classes=3, gamma=0.1)

    assert np.allclose(out1, out2)
    assert out1.shape == (6, 3)
