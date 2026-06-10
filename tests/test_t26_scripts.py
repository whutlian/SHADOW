from pathlib import Path

from scripts.run_t26_arxiv_teacher_sweep import build_rows as build_arxiv_rows
from scripts.run_t26_products_recovery import build_products_outputs
from scripts.run_t26_reddit_trainer_sweep import build_rows as build_reddit_rows
from scripts.run_t26_stage import REQUIRED_OUTPUTS, _build_requirement_checks
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


def test_t26_reddit_seed_sweep_stats_written_after_all_five_seeds(tmp_path: Path):
    source = tmp_path / "reddit.csv"
    lines = ["dataset,method,seed,requested_full_node_ratio,accuracy,macro_f1,predicted_class_count"]
    for seed in range(1, 6):
        lines.append(f"Reddit,sft_hnr_fdm_hybrid,{seed},0.005,0.93,0.89,41")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = build_reddit_rows(seed=42, t25_csv=source)
    method_rows = [row for row in rows if row["method"] == "reddit_sft_hnr_fdm_hybrid" and float(row["requested_full_node_ratio"]) == 0.005]

    assert all(row["accuracy"] != "" for row in method_rows)
    assert all(row.get("seed_sweep_mean_acc", "") == 0.93 for row in method_rows)


def test_t26_products_long_results_unblock_products_rows(tmp_path: Path):
    long_csv = tmp_path / "products_long.csv"
    t25_csv = tmp_path / "t25.csv"
    t25_csv.write_text(
        "dataset,method,requested_full_node_ratio,accuracy,macro_f1,predicted_class_count,products_diag_memmap_row_order_matches_node_id,products_diag_masks_aligned\n"
        "ogbn-products,P0_identity_replay,0.0025,0.75,0.4,47,True,True\n",
        encoding="utf-8",
    )
    long_csv.write_text(
        "dataset,method,requested_full_node_ratio,accuracy,macro_f1,predicted_class_count,status,p0a_alltrain_acc,p0b_self_fit_acc,p0d_prototype_oracle_acc,p0d_centroid_oracle_acc\n"
        "ogbn-products,P0a_alltrain_condensed_trainer_parity,0.08,0.75,0.40,47,completed_long,0.75,,,,\n"
        "ogbn-products,P0b_selected_prototype_self_fit,0.0025,0.96,0.80,47,completed_long,,0.96,,,\n"
        "ogbn-products,products_uca_hybrid_balanced_trainer,0.0025,0.71,0.30,46,completed_long,,,,\n",
        encoding="utf-8",
    )

    class Args:
        products_root = "missing"
        t25_products_csv = t25_csv
        long_results_csv = long_csv
        ratios = [0.0025]
        uca_domains = 8
        seed = 42
        per_class_csv = tmp_path / "per_class.csv"

    diagnostics, uca_rows, _per_class = build_products_outputs(Args())
    p0a = next(row for row in diagnostics if row["method"] == "P0a_alltrain_condensed_trainer_parity")
    p0b = next(row for row in diagnostics if row["method"] == "P0b_selected_prototype_self_fit")
    uca = next(row for row in uca_rows if row["method"] == "products_uca_hybrid_balanced_trainer")

    assert p0a["p0a_passed"] is True
    assert p0b["p0b_passed"] is True
    assert uca["status"] == "completed_long"
    assert uca["accuracy"] == 0.71


def test_t26_stage_checks_mark_products_p0_complete_after_long_pass():
    rows = [
        {
            "dataset": "ogbn-products",
            "method": "P0a_alltrain_condensed_trainer_parity",
            "status": "completed_long",
            "p0a_passed": True,
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "ogbn-products",
            "method": "P0b_selected_prototype_self_fit",
            "requested_full_node_ratio": 0.0025,
            "status": "completed_long",
            "p0b_passed": True,
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "ogbn-products",
            "method": "P0b_selected_prototype_self_fit",
            "requested_full_node_ratio": 0.005,
            "status": "completed_long",
            "p0b_passed": True,
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "Reddit",
            "method": "reddit_sft_hnr_fdm_hybrid",
            "status": "completed_reuse_existing_t25_seed",
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "ogbn-arxiv",
            "method": "arxiv_teacher_sweep",
            "condensation_status": "blocked_by_teacher_gate",
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "MAG240M",
            "method": "t26_ultra_contract_regression",
            "promotion_status": "not_promoted",
        },
    ]

    checks = {row["requirement_check"]: row["requirement_status"] for row in _build_requirement_checks(rows)}

    assert checks["products_P0a"] == "completed"
    assert checks["products_P0b"] == "completed"


def test_t26_stage_checks_block_partial_p0b_ratio_coverage():
    rows = [
        {
            "dataset": "ogbn-products",
            "method": "P0a_alltrain_condensed_trainer_parity",
            "status": "completed_long",
            "p0a_passed": True,
            "promotion_status": "not_promoted",
        },
        {
            "dataset": "ogbn-products",
            "method": "P0b_selected_prototype_self_fit",
            "requested_full_node_ratio": 0.0025,
            "status": "completed_long",
            "p0b_passed": True,
            "promotion_status": "not_promoted",
        },
    ]

    checks = {row["requirement_check"]: row["requirement_status"] for row in _build_requirement_checks(rows)}

    assert checks["products_P0b"] == "blocked"


def test_t26_stage_checks_block_partial_reddit_seed_grid():
    rows = []
    for method in [
        "reddit_current_sft_signature_random",
        "reddit_current_sft_signature_medoid",
        "reddit_current_sft_signature_kcenter",
        "reddit_sft_hnr_fdm_hybrid",
    ]:
        rows.append(
            {
                "dataset": "Reddit",
                "method": method,
                "requested_full_node_ratio": 0.005,
                "seed": 1,
                "status": "completed_reuse_existing_t25_seed",
                "accuracy": 0.93,
                "promotion_status": "not_promoted",
            }
        )

    checks = {row["requirement_check"]: row["requirement_status"] for row in _build_requirement_checks(rows)}

    assert checks["reddit_seed_sweep"] == "blocked"
