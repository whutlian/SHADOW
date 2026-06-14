from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_scr_bank import (
    SCR_POLICY_CLASS_RANDOM,
    SCR_POLICY_FULL,
    SCR_POLICY_FULL_TEACHER_WEIGHT,
    build_scr_rank_from_arrays,
)
from shadow_hgc.ultra.papers100m_scr_materialize import audit_scr_prefixes


def _toy_arrays():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int16)
    train_mask = np.ones(labels.shape[0], dtype=bool)
    valid_labels = np.array([2, 2, 1, 0], dtype=np.int16)
    test_labels = np.array([1, 0, 2, 1], dtype=np.int16)
    degree = np.array([0, 1, 4, 2, 7, 15, 1, 3, 8, 31, 63, 127], dtype=np.float32)
    features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.8, 0.2],
            [0.6, 0.0, 0.4, 0.0],
            [0.0, 0.6, 0.0, 0.4],
            [0.4, 0.0, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    teacher_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.uint16)
    teacher_conf = np.array([0.05, 0.10, 0.99, 0.05, 0.10, 0.99, 0.05, 0.10, 0.99, 0.95, 0.95, 0.95], dtype=np.float32)
    return labels, train_mask, valid_labels, test_labels, degree, features, teacher_pred, teacher_conf


def test_scr_class_random_respects_class_floors_and_nested_prefixes():
    labels, train_mask, _valid, _test, degree, features, teacher_pred, teacher_conf = _toy_arrays()

    result = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_CLASS_RANDOM,
        seed=42,
        max_rows=9,
        num_classes=3,
        class_floor_requested=2,
    )
    audit = audit_scr_prefixes(result.global_rank, ratios=[1 / 12, 3 / 12, 6 / 12], denominator=12)

    assert result.global_rank.shape[0] == 9
    assert min(np.bincount(labels[result.global_rank[:6]], minlength=3)) >= 2
    assert all(int(row["prefix_violation_count"]) == 0 for row in audit)


def test_scr_rank_is_deterministic_for_seed_and_changes_for_other_seed():
    labels, train_mask, _valid, _test, degree, features, teacher_pred, teacher_conf = _toy_arrays()

    first = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL,
        seed=7,
        max_rows=8,
        num_classes=3,
    )
    second = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL,
        seed=7,
        max_rows=8,
        num_classes=3,
    )
    other = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL,
        seed=8,
        max_rows=8,
        num_classes=3,
    )

    assert np.array_equal(first.global_rank, second.global_rank)
    assert not np.array_equal(first.global_rank, other.global_rank)


def test_scr_full_does_not_let_teacher_confidence_replace_class_coverage():
    labels, train_mask, _valid, _test, degree, features, teacher_pred, teacher_conf = _toy_arrays()

    result = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL,
        seed=42,
        max_rows=6,
        num_classes=3,
        class_floor_requested=1,
    )

    selected_labels = labels[result.global_rank]
    assert set(selected_labels.tolist()) == {0, 1, 2}
    assert np.any(teacher_conf[result.global_rank] < 0.5)


def test_scr_teacher_weight_variant_uses_capped_mild_tiebreak():
    labels, train_mask, _valid, _test, degree, features, teacher_pred, teacher_conf = _toy_arrays()

    unweighted = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL,
        seed=4,
        max_rows=8,
        num_classes=3,
    )
    weighted = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        policy=SCR_POLICY_FULL_TEACHER_WEIGHT,
        seed=4,
        max_rows=8,
        num_classes=3,
        teacher_weight_eta=0.2,
    )

    assert np.all(weighted.teacher_tiebreak_weight >= 0.5)
    assert np.all(weighted.teacher_tiebreak_weight <= 2.0)
    assert set(labels[weighted.global_rank[:6]].tolist()) == {0, 1, 2}
    assert not np.array_equal(unweighted.priority, weighted.priority)


def test_scr_selection_ignores_valid_and_test_labels():
    labels, train_mask, valid_labels, test_labels, degree, features, teacher_pred, teacher_conf = _toy_arrays()

    first = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        valid_labels=valid_labels,
        test_labels=test_labels,
        policy=SCR_POLICY_FULL,
        seed=11,
        max_rows=8,
        num_classes=3,
    )
    second = build_scr_rank_from_arrays(
        labels=labels,
        candidate_mask=train_mask,
        degree=degree,
        features=features,
        teacher_pred_class=teacher_pred,
        teacher_confidence=teacher_conf,
        valid_labels=valid_labels[::-1] + 10,
        test_labels=test_labels[::-1] + 10,
        policy=SCR_POLICY_FULL,
        seed=11,
        max_rows=8,
        num_classes=3,
    )

    assert np.array_equal(first.global_rank, second.global_rank)
    assert np.array_equal(first.coverage_bucket, second.coverage_bucket)
