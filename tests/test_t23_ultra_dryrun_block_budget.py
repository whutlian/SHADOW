from scripts.dry_run_t23_ultra_sft import build_rows


def test_t23_ultra_dryrun_marks_all_target_ultra_cache_forbidden():
    rows = build_rows()
    paper_all = next(row for row in rows if row["dataset"] == "ogbn-papers100M" and row["cache_mode"] == "all_target_rows")
    paper_train = next(row for row in rows if row["dataset"] == "ogbn-papers100M" and row["cache_mode"] == "train_target_only")
    assert paper_all["wall_time_category"] == "forbidden_by_t23_ultra_policy"
    assert paper_train["ultra_policy"] == "train_target_only_required"
    assert paper_train["uses_e_by_d_materialization"] is False
