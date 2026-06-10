from pathlib import Path

from scripts.run_t26_arxiv_teacher_sweep import build_rows as build_arxiv_rows
from scripts.run_t26_reddit_trainer_sweep import build_rows as build_reddit_rows
from scripts.run_t26_stage import REQUIRED_OUTPUTS
from scripts.run_t26_ultra_contract_regression import build_rows as build_ultra_rows


def test_t26_required_outputs_are_declared():
    for path in [
        "experiments/tables/t26_stage_summary_seed42.csv",
        "experiments/summaries/t26_stage_summary.md",
        "experiments/tables/t26_products_recovery_diagnostics_seed42.csv",
        "experiments/tables/t26_products_uca_sweep_seed42.csv",
        "experiments/tables/t26_reddit_seed_trainer_mixup_sweep.csv",
        "experiments/tables/t26_arxiv_teacher_sweep_seed42.csv",
        "experiments/tables/t26_ultra_contract_regression_seed42.csv",
    ]:
        assert Path(path) in REQUIRED_OUTPUTS


def test_t26_arxiv_teacher_rows_block_condensation_until_a1():
    rows = build_arxiv_rows(seed=42)

    assert rows
    assert all(row["stage"] == "t26" for row in rows)
    assert any(row["condensation_status"] == "blocked_by_teacher_gate" for row in rows)
    assert all(float(row["requested_full_node_ratio"]) == float(row["actual_full_node_ratio"]) for row in rows)


def test_t26_ultra_rows_keep_forbidden_flags_false():
    rows = build_ultra_rows(seed=42)

    assert rows
    for row in rows:
        assert row["stage"] == "t26"
        assert row["uses_all_target_cache"] is False
        assert row["uses_exact_pairwise"] is False
        assert row["uses_e_by_d_materialization"] is False


def test_t26_reddit_seed_sweep_stats_require_all_five_seeds(tmp_path: Path):
    source = tmp_path / "reddit.csv"
    source.write_text(
        "dataset,method,seed,requested_full_node_ratio,accuracy,macro_f1,predicted_class_count\n"
        "Reddit,sft_hnr_fdm_hybrid,1,0.005,0.93,0.89,41\n",
        encoding="utf-8",
    )

    rows = build_reddit_rows(seed=42, t25_csv=source)
    method_rows = [row for row in rows if row["method"] == "reddit_sft_hnr_fdm_hybrid" and float(row["requested_full_node_ratio"]) == 0.005]

    assert any(row["accuracy"] != "" for row in method_rows)
    assert any(row["status"] == "ready_not_run" for row in method_rows)
    assert all(row.get("seed_sweep_mean_acc", "") == "" for row in method_rows)
