from __future__ import annotations

from scripts.run_t31_stage import build_stage_summary_rows


def test_t31_stage_summary_reports_required_checks() -> None:
    rows = build_stage_summary_rows(
        reddit_ttc=[],
        reddit_simsft=[],
        reddit_bonsai=[],
        qoc_forensic=[],
        arxiv_cns=[],
        arxiv_semantic=[],
        products=[],
    )
    checks = {row["requirement_check"]: row for row in rows}
    for key in [
        "reddit_ttc_rows_present",
        "reddit_ttc_real_metrics_present",
        "reddit_ttc_0p10_gate",
        "reddit_ttc_0p50_gate",
        "qoc_forensic_identity_sanity",
        "arxiv_base_logits_present",
        "arxiv_semantic_cache_present",
        "products_maintenance_multiseed",
        "forbidden_guard_hits",
        "promoted_safe_rows",
        "promoted_sota_chase_rows",
    ]:
        assert key in checks
