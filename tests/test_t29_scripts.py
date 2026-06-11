from __future__ import annotations

from pathlib import Path

from scripts.run_t29_arxiv_cns_actual import build_arxiv_cns_rows
from scripts.run_t29_arxiv_semantic_teacher import build_semantic_rows
from scripts.run_t29_products_maintenance import build_products_rows
from scripts.run_t29_reddit_control_audit import build_control_rows
from scripts.run_t29_reddit_operator_match import build_omcp_rows
from scripts.run_t29_stage import build_stage_summary_rows
from shadow_hgc.sft.t29_contract import T29_REQUIRED_FIELDS


def _args(**kwargs):
    return type("Args", (), kwargs)()


def test_t29_all_script_rows_have_required_fields(tmp_path: Path):
    rows = []
    rows += build_arxiv_cns_rows(_args(seed=42, base_predictors=["raw_x_mlp"], base_logits_dir=str(tmp_path), smoke=False))
    rows += build_semantic_rows(_args(seed=42, lm_models=["specter"], semantic_cache_dir=str(tmp_path), raw_text_path=str(tmp_path / "missing.jsonl"), use_precomputed_semantic_features="", smoke=False))
    rows += build_control_rows(_args(seed=42, seeds=[42], ratios=[0.001, 0.005], methods=["current_sft_signature_random"], smoke=True))
    rows += build_omcp_rows(_args(seed=42, ratios=[0.001], prototype_inits=["current_sft_signature_random"], operator_topks=[4], students=["operator_sft_table_head"], smoke=True))
    rows += build_products_rows(_args(seed=42, ratios=[0.0002, 0.0025], seeds=[42], smoke=True))
    for row in rows:
        assert set(T29_REQUIRED_FIELDS).issubset(row)


def test_t29_control_rows_scale_budget_and_import_references():
    rows = build_control_rows(_args(seed=42, seeds=[42], ratios=[0.001, 0.005], methods=["current_sft_signature_random"], smoke=True))
    by_ratio = {row["requested_full_node_ratio"]: row for row in rows}
    assert by_ratio[0.001]["actual_condensed_nodes"] == 233
    assert by_ratio[0.005]["actual_condensed_nodes"] == 1165
    assert by_ratio[0.005]["accuracy"] != ""


def test_t29_stage_summary_blocks_missing_tables(tmp_path: Path):
    rows = build_stage_summary_rows(
        arxiv_cns_csv=tmp_path / "missing_arxiv.csv",
        semantic_csv=tmp_path / "missing_semantic.csv",
        reddit_control_csv=tmp_path / "missing_control.csv",
        omcp_csv=tmp_path / "missing_omcp.csv",
        pltc_csv=tmp_path / "missing_pltc.csv",
        bonsai_csv=tmp_path / "missing_bonsai.csv",
        products_csv=tmp_path / "missing_products.csv",
    )
    statuses = {row["requirement_check"]: row["requirement_status"] for row in rows}
    assert statuses["arxiv_actual_cns_base_logits"] == "blocked"
    assert statuses["reddit_omcp_improves_0p10_or_0p50"] == "blocked"
    assert statuses["promoted_forbidden_guard"] == "completed"
