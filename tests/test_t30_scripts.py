from __future__ import annotations

from argparse import Namespace

from scripts.run_t30_arxiv_qoc import build_arxiv_qoc_rows
from scripts.run_t30_products_maintenance import build_products_rows
from scripts.run_t30_reddit_qoc import build_reddit_qoc_rows
from scripts.run_t30_stage import build_stage_summary_rows
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, validate_t30_row


def test_t30_reddit_qoc_rows_have_required_schema_and_block_without_real_cache() -> None:
    rows = build_reddit_qoc_rows(
        Namespace(
            seed=42,
            ratios=[0.001, 0.005],
            assignment_modes=["qoc_class_conditional_online_kmeans"],
            operator_topks=[8],
            quotient_build_modes=["code_row_normalized_fallback"],
            students=["operator_sft_table_head"],
            hidden_dims=[128],
            epochs=[60],
            enable_pltc=False,
            promotion_track="safe_main",
            run_long=False,
            smoke=True,
            sft_cache_dir="missing",
        )
    )
    assert {int(row["num_codewords"]) for row in rows} == {233, 1165}
    for row in rows:
        assert set(T30_REQUIRED_FIELDS).issubset(row)
        assert row["transfer_eval_type"] != "real_transfer_eval"
        assert row["status"] in {"completed_operator_smoke", "blocked"}
        assert validate_t30_row(row)["valid"]


def test_t30_arxiv_qoc_blocks_until_teacher_gate_passes() -> None:
    rows = build_arxiv_qoc_rows(Namespace(seed=42, ratios=[0.005], teacher_cache="", run_long=False))
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "teacher_gate_not_passed"


def test_t30_products_rows_are_maintenance_not_promotions() -> None:
    rows = build_products_rows(Namespace(seed=42, seeds=[42], ratios=[0.0002, 0.005], methods=["products_uca_hybrid_mixup"]))
    assert len(rows) == 2
    assert all(row["promotion_status"] == "not_promoted" for row in rows)
    assert all(row["status"] == "carried_forward_reference" for row in rows)


def test_t30_stage_summary_reports_required_checks() -> None:
    rows = build_stage_summary_rows(
        arxiv_cns=[],
        semantic=[],
        reddit_qoc=[],
        arxiv_qoc=[],
        products=[],
    )
    checks = {row["requirement_check"]: row for row in rows}
    for key in [
        "promoted_safe_rows",
        "promoted_sota_chase_rows",
        "blocked_rows_by_reason",
        "best_reddit_0p10",
        "best_reddit_0p50",
        "best_arxiv_teacher",
        "arxiv_A1_A2_A3_status",
        "products_maintenance_status",
        "forbidden_guard_hits",
    ]:
        assert key in checks
