from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_t28_arxiv_teacher_pivot import build_arxiv_rows
from scripts.run_t28_products_maintenance import build_products_rows
from scripts.run_t28_reddit_control_audit import build_control_rows
from scripts.run_t28_reddit_structure import build_structure_rows
from scripts.run_t28_stage import build_stage_summary_rows


def test_t28_arxiv_smoke_rows_include_cns_flags():
    args = argparse.Namespace(seed=42, smoke=True, run_long=False, enable_cns=True)
    rows = build_arxiv_rows(args)
    assert any(row["uses_cns_postprocess"] is True for row in rows)
    assert all(row["uses_teacher_logits_for_condensation"] is False for row in rows)
    assert all(row["uses_valid_labels_as_input"] is False for row in rows)
    assert all(row["uses_test_labels_as_input"] is False for row in rows)


def test_t28_reddit_control_smoke_has_required_ratios_and_no_graph_flags():
    args = argparse.Namespace(ratios=[0.0005, 0.001], methods=["current_sft_signature_random", "reddit_random_frozen_init"], seeds=[42], smoke=True)
    rows = build_control_rows(args)
    assert len(rows) == 4
    assert {row["requested_full_node_ratio"] for row in rows} == {0.0005, 0.001}
    assert all(row["edge_builder"] == "table_only" for row in rows)
    assert all(row["uses_processed_data_pt"] is False for row in rows)


def test_t28_reddit_structure_smoke_declares_graph_builders():
    args = argparse.Namespace(ratios=[0.001], edge_builders=["knn", "cooccur", "edge_predictor"], prototype_inits=["current_sft_signature_random"], edge_topks=[4], students=["weighted_gcn"], seed=42, smoke=True)
    rows = build_structure_rows(args)
    assert {row["edge_builder"] for row in rows} == {"knn", "cooccur", "edge_predictor"}
    assert all(row["edge_weight_normalization"] == "dst_row" for row in rows)
    assert all(row["uses_processed_data_pt"] is False for row in rows)


def test_t28_products_maintenance_smoke_retains_known_ratios():
    args = argparse.Namespace(ratios=[0.0002, 0.0004, 0.0008, 0.0025, 0.005], seed=42, smoke=True)
    rows = build_products_rows(args)
    assert {row["requested_full_node_ratio"] for row in rows} == {0.0002, 0.0004, 0.0008, 0.0025, 0.005}
    assert all(row["method"] == "products_uca_hybrid_mixup" for row in rows)


def test_t28_stage_summary_blocks_missing_tables(tmp_path: Path):
    rows = build_stage_summary_rows(
        arxiv_csv=tmp_path / "missing_arxiv.csv",
        reddit_control_csv=tmp_path / "missing_control.csv",
        reddit_structure_csv=tmp_path / "missing_structure.csv",
        products_csv=tmp_path / "missing_products.csv",
    )
    checks = {row["requirement_check"]: row["requirement_status"] for row in rows}
    assert checks["arxiv_A1_gate"] == "blocked"
    assert checks["reddit_forbidden_flags"] == "completed"
    assert checks["products_maintenance"] == "blocked"
