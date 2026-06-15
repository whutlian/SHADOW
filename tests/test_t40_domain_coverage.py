from __future__ import annotations

import numpy as np
import torch

from shadow_hgc.sft.domain_coverage import (
    build_domain_bucket_ids,
    domain_coverage_gap,
    domain_train_all_undercoverage_scores,
    domain_undercoverage_scores,
)


def test_t40_domain_buckets_ignore_valid_and_test_labels() -> None:
    signatures = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ],
        dtype=np.float32,
    )
    train_mask = np.asarray([True, True, False, False, False, False])
    labels_a = torch.tensor([0, 0, 1, 1, 2, 2])
    labels_b = torch.tensor([0, 0, 2, 2, 1, 1])

    buckets_a = build_domain_bucket_ids(
        signatures,
        train_mask=train_mask,
        labels=labels_a,
        seed=7,
        projection_dim=2,
        num_quantiles=2,
    )
    buckets_b = build_domain_bucket_ids(
        signatures,
        train_mask=train_mask,
        labels=labels_b,
        seed=7,
        projection_dim=2,
        num_quantiles=2,
    )

    assert buckets_a.tolist() == buckets_b.tolist()
    assert domain_coverage_gap(buckets_a, [0, 1]) == domain_coverage_gap(buckets_b, [0, 1])


def test_t40_domain_undercoverage_prioritizes_missing_all_target_buckets() -> None:
    buckets = np.asarray([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    selected_auto = [0, 1, 2, 3]
    scores = domain_undercoverage_scores(buckets, selected_auto)

    assert scores[4] > scores[0]
    assert scores[6] > scores[2]

    domain_selected = selected_auto + [int(np.argmax(scores))]

    assert domain_coverage_gap(buckets, domain_selected) < domain_coverage_gap(buckets, selected_auto)


def test_t40_domain_score_uses_train_vs_all_undercoverage() -> None:
    all_buckets = np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    train_rows = np.asarray([0, 1, 2, 3], dtype=np.int64)
    scores = domain_train_all_undercoverage_scores(all_buckets, train_rows)

    assert scores[3] > scores[0]
    assert scores[3] == scores[4]
