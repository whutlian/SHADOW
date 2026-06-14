from __future__ import annotations

import numpy as np

from shadow_hgc.ultra.papers100m_ant import materialize_ant_edges, train_or_load_ant_link_predictor
from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest
from shadow_hgc.ultra.papers100m_nested_bank import NESTED_BANK_POLICY, audit_nested_bank, build_nested_bank_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sft_cache import build_or_load_sft_cache
from shadow_hgc.ultra.papers100m_sgc_backend import weighted_sgc_propagate
from shadow_hgc.ultra.papers100m_stt_bank import load_selection_bank
from shadow_hgc.ultra.papers100m_teacher import write_teacher_topk_cache_from_probs

from t35_fixtures import make_toy_papers100m_root


def _toy_cache(tmp_path):
    data_root = make_toy_papers100m_root(tmp_path)
    cache_root = tmp_path / "cache"
    build_papers100m_manifest(data_root, cache_root, allow_toy=True)
    build_or_load_edge_slice_cache(cache_root, data_root=data_root, chunk_size_edges=2, force=True)
    build_or_load_sft_cache(cache_root, chunk_size_edges=2, force=True)
    write_teacher_topk_cache_from_probs(
        cache_root,
        np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.15], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]], dtype=np.float32),
        mode="topk2_tail",
    )
    return cache_root


def test_t36_nested_bank_prefixes_are_t35_materializer_compatible(tmp_path):
    cache_root = _toy_cache(tmp_path)
    ctx = Papers100MCacheContext(cache_root, selection_policy=NESTED_BANK_POLICY, seed=7)
    manifest = build_nested_bank_v2(ctx, max_ratio=0.75, seed=7, force=True)
    bank = load_selection_bank(cache_root, policy=NESTED_BANK_POLICY, seed=7)

    small = bank.select_prefix(0.25, full_node_denominator=4)
    large = bank.select_prefix(0.75, full_node_denominator=4)
    audit = audit_nested_bank(cache_root, policy=NESTED_BANK_POLICY, seed=7, ratios=[0.25, 0.50, 0.75])

    assert manifest["nested_bank_id"] == manifest["selection_bank_id"]
    assert small.size == 1
    assert large.size == 3
    assert set(small.tolist()).issubset(set(large.tolist()))
    assert all(int(row["prefix_violation_count"]) == 0 for row in audit)


def test_t36_weighted_sgc_matches_explicit_destination_normalized_scatter():
    x = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([2, 2], dtype=np.int64)
    weight = np.array([1.0, 3.0], dtype=np.float32)

    out = weighted_sgc_propagate(x, src, dst, weight, num_hops=1, normalize_dst=True, add_self=False)

    expected = np.zeros_like(x)
    expected[2] = x[0] * 0.25 + x[1] * 0.75
    assert np.allclose(out, expected)


def test_t36_ant_materialization_is_bounded_and_reuses_link_predictor(tmp_path):
    cache_root = _toy_cache(tmp_path)
    ctx = Papers100MCacheContext(cache_root, selection_policy=NESTED_BANK_POLICY, seed=7)
    bank_manifest = build_nested_bank_v2(ctx, max_ratio=0.75, seed=7, force=True)
    link = train_or_load_ant_link_predictor(cache_root, nested_bank_id=bank_manifest["nested_bank_id"], teacher_id="toy", seed=7)
    link_again = train_or_load_ant_link_predictor(cache_root, nested_bank_id=bank_manifest["nested_bank_id"], teacher_id="toy", seed=7)
    ant = materialize_ant_edges(
        cache_root,
        policy=NESTED_BANK_POLICY,
        seed=7,
        ratio=0.75,
        edge_topk=2,
        link_predictor_id=link["ant_link_predictor_id"],
        candidate_multiplier=3,
        force=True,
    )

    assert link["ant_link_predictor_id"] == link_again["ant_link_predictor_id"]
    assert ant["ant_bounded"] is True
    assert ant["ant_candidate_count"] <= ant["candidate_bound"]
    assert ant["edge_weight_nonnegative"] is True
    assert ant["uses_exact_all_pair_distance"] is False
