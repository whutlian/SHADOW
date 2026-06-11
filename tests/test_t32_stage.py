from __future__ import annotations

from scripts.run_t32_stage import build_stage_summary_rows


def test_t32_stage_summary_reports_required_checks() -> None:
    rows = build_stage_summary_rows(
        reddit_ttcpp=[],
        reddit_multiseed=[],
        teacher_ensemble=[],
        arxiv_cns=[],
        arxiv_semantic=[],
        products=[],
    )
    checks = {row["requirement_check"]: row for row in rows}
    for key in [
        "reddit_ttcpp_rows_present",
        "reddit_ttcpp_0p10_recovered",
        "reddit_ttcpp_0p50_first_gate",
        "reddit_ttcpp_0p50_main_gate",
        "teacher_ensemble_cache_present",
        "arxiv_raw_mlp_cns_sanity",
        "arxiv_sft_cns_gate",
        "arxiv_semantic_cache_present",
        "products_maintenance_multiseed",
        "forbidden_guard_hits",
        "promoted_sota_chase_rows",
    ]:
        assert key in checks
