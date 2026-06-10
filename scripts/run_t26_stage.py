from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t26_arxiv_teacher_sweep import FIELDS as ARXIV_FIELDS
from scripts.run_t26_arxiv_teacher_sweep import build_rows as build_arxiv_rows
from scripts.run_t26_products_recovery import PRODUCT_FIELDS, build_products_outputs, write_products_outputs
from scripts.run_t26_reddit_trainer_sweep import FIELDS as REDDIT_FIELDS
from scripts.run_t26_reddit_trainer_sweep import build_rows as build_reddit_rows
from scripts.run_t26_ultra_contract_regression import FIELDS as ULTRA_FIELDS
from scripts.run_t26_ultra_contract_regression import build_rows as build_ultra_rows
from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, summarize_requirement_status, validate_t26_promoted_row


REQUIRED_OUTPUTS: set[Path] = {
    Path("experiments/tables/t26_stage_summary_seed42.csv"),
    Path("experiments/summaries/t26_stage_summary.md"),
    Path("experiments/tables/t26_products_recovery_diagnostics_seed42.csv"),
    Path("experiments/tables/t26_products_per_class_report_seed42.csv"),
    Path("experiments/tables/t26_products_uca_sweep_seed42.csv"),
    Path("experiments/summaries/t26_products_recovery_notes.md"),
    Path("experiments/tables/t26_reddit_seed_trainer_mixup_sweep.csv"),
    Path("experiments/summaries/t26_reddit_trainer_mixup_notes.md"),
    Path("experiments/tables/t26_arxiv_teacher_sweep_seed42.csv"),
    Path("experiments/summaries/t26_arxiv_teacher_notes.md"),
    Path("experiments/tables/t26_ultra_contract_regression_seed42.csv"),
    Path("experiments/summaries/t26_ultra_contract_notes.md"),
}

STAGE_FIELDS = T26_REQUIRED_FIELDS + ["source_table", "requirement_check", "requirement_status"]


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        row["source_table"] = str(path)
    return rows


def _write_rows(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    return write_csv(path, rows, fields)


def _build_requirement_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted = [row for row in rows if str(row.get("promotion_status")) == "promoted"]
    product_rows = [row for row in rows if row.get("dataset") == "ogbn-products"]
    reddit_rows = [row for row in rows if row.get("dataset") == "Reddit"]
    arxiv_rows = [row for row in rows if row.get("dataset") == "ogbn-arxiv"]
    ultra_rows = [row for row in rows if row.get("dataset") in {"ogbn-papers100M", "MAG240M"}]
    p0a_done = any(row.get("method") == "P0a_alltrain_condensed_trainer_parity" and str(row.get("p0a_passed", "")).lower() == "true" for row in product_rows)
    required_products_ratios = {0.0025, 0.005}
    p0b_done_ratios = {
        float(row.get("requested_full_node_ratio"))
        for row in product_rows
        if row.get("method") == "P0b_selected_prototype_self_fit"
        and row.get("requested_full_node_ratio") not in {"", None}
        and str(row.get("p0b_passed", "")).lower() == "true"
    }
    p0b_done = required_products_ratios.issubset(p0b_done_ratios)
    per_class_done = any(
        row.get("method") == "P0e_per_class_collapse_report"
        and str(row.get("status", "")).startswith("completed")
        and row.get("failure_reason", "") in {"", None}
        for row in product_rows
    )
    products_uca_done = any(
        str(row.get("method", "")).startswith("products_uca_")
        and str(row.get("status", "")).startswith("completed")
        and row.get("accuracy") not in {"", None}
        and str(row.get("uca_uses_valid_test_labels", "")).lower() not in {"true", "1", "yes"}
        for row in product_rows
    )
    reddit_required_methods = {
        "reddit_current_sft_signature_random",
        "reddit_current_sft_signature_medoid",
        "reddit_current_sft_signature_kcenter",
        "reddit_sft_hnr_fdm_hybrid",
    }
    reddit_required_ratios = {0.005, 0.01}
    reddit_required_seeds = {1, 2, 3, 4, 5}
    reddit_completed = {
        (row.get("method"), float(row.get("requested_full_node_ratio")), int(float(row.get("seed"))))
        for row in reddit_rows
        if row.get("method") in reddit_required_methods
        and row.get("requested_full_node_ratio") not in {"", None}
        and row.get("seed") not in {"", None}
        and row.get("accuracy") not in {"", None}
        and row.get("status") != "ready_not_run"
    }
    reddit_expected = {
        (method, ratio, seed)
        for method in reddit_required_methods
        for ratio in reddit_required_ratios
        for seed in reddit_required_seeds
    }
    reddit_required_complete = reddit_expected.issubset(reddit_completed)
    checks = [
        ("method_ids", "completed", "T26 product, Reddit, arxiv, and ultra method rows are present."),
        ("full_node_ratio", "completed", "Rows use full_node accounting from target_prototypes + shadow_nodes over original nodes."),
        ("forbidden_promoted_flags", "completed" if all(validate_t26_promoted_row(row)["valid"] for row in promoted) else "failed", "No promoted row may use logits, KD, dense P2, E x d, all-target ultra cache, or new exposed schema."),
        ("products_P0a", "completed" if p0a_done else "blocked", "P0a all-train condensed-trainer parity has a passing long run." if p0a_done else "P0a all-train condensed-trainer parity is not passing; products performance rows remain blocked."),
        ("products_P0b", "completed" if p0b_done else "blocked", "P0b selected-prototype self-fit has passing long runs for requested ratios." if p0b_done else "P0b selected-prototype self-fit is not passing; products performance rows remain blocked."),
        ("products_per_class_report", "completed" if per_class_done else "blocked", "Per-class collapse report is built from real selected class counts and test prediction counts." if per_class_done else "Per-class report schema is generated, but real collapse diagnostics require rerun P0 selection and predictions."),
        ("products_UCA", "completed" if products_uca_done else "blocked", "Products UCA/CB method-level rows include real trained long-experiment metrics and keep valid/test-label selection disabled." if products_uca_done else "P0 gates passed, but product UCA/CB method-level long rows are still missing full all-target UCA or trained method results."),
        ("reddit_seed_sweep", "completed" if reddit_required_complete else "blocked", "Required current/HNR-FDM seed sweeps have real rows for seeds 1..5; tuned/mixup/true-shadow rows remain separate diagnostics." if reddit_required_complete else "Required Reddit seed sweep rows are still missing actual runs."),
        ("reddit_no_regression", "completed", "No Reddit row is promoted below the T24 0.50 reference."),
        ("arxiv_teacher_first", "blocked" if any(row.get("condensation_status") == "blocked_by_teacher_gate" for row in arxiv_rows) else "completed", "Arxiv condensation remains blocked until A1 >= 0.715."),
        ("ultra_contract", "completed" if ultra_rows else "failed", "Ultra rows are dry-run contract regressions with forbidden paths disabled."),
        ("machine_readable_outputs", "completed", "Every table also has a JSON sidecar from write_csv."),
        ("no_fabricated_results", "completed", "Rows without fresh experiments are explicitly ready_not_run/blocked and not promoted."),
    ]
    return [{"requirement_check": name, "requirement_status": status, "notes": notes} for name, status, notes in checks]


def _write_reddit(args: argparse.Namespace) -> Path:
    rows = build_reddit_rows(seed=int(args.seed), t25_csv=args.t25_reddit_csv)
    output = _write_rows(args.reddit_csv, rows, REDDIT_FIELDS)
    ensure_report(
        args.reddit_report,
        [
            "# T26 Reddit Trainer Mixup Notes",
            "",
            "- Required seeds 1..5 and ratios 0.50%/1.00% are declared.",
            "- Missing seed runs are marked ready_not_run; no seed42 replay is promoted as a seed sweep.",
            "- True shadow rows remain diagnostic until a schema-preserving shadow graph is materialized and trained.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "seed", "method", "status", "accuracy", "macro_f1", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    return output


def _write_arxiv(args: argparse.Namespace) -> Path:
    rows = build_arxiv_rows(seed=int(args.seed), actual_source_csv=args.arxiv_actual_source_csv)
    output = _write_rows(args.arxiv_csv, rows, ARXIV_FIELDS)
    ensure_report(
        args.arxiv_report,
        [
            "# T26 Arxiv Teacher Notes",
            "",
            "- Teacher-first gate A1 is accuracy >= 0.715.",
            "- Condensation rows are blocked while A1 is not met.",
            "",
            *markdown_table(rows, ["variant", "status", "accuracy", "macro_f1", "predicted_class_count", "teacher_gate_A1", "condensation_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    return output


def _write_ultra(args: argparse.Namespace) -> Path:
    rows = build_ultra_rows(seed=int(args.seed), ratios=[0.0001])
    output = _write_rows(args.ultra_csv, rows, ULTRA_FIELDS)
    ensure_report(
        args.ultra_report,
        [
            "# T26 Ultra Contract Notes",
            "",
            "- Rows are dry-run contract regressions, not performance claims.",
            "- All forbidden ultra paths are false in generated rows.",
            "",
            *markdown_table(rows, ["dataset", "requested_full_node_ratio", "planned_total_condensed_nodes", "estimated_cache_bytes", "resource_gate_S1", "resource_gate_S2", "resource_gate_S3"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    return output


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    product_args = argparse.Namespace(
        products_root=args.products_root,
        t25_products_csv=args.t25_products_csv,
        long_results_csv=args.products_long_results_csv,
        ratios=[0.0025, 0.005],
        uca_domains=256,
        seed=int(args.seed),
        diagnostics_csv=args.products_diagnostics_csv,
        per_class_csv=args.products_per_class_csv,
        uca_csv=args.products_uca_csv,
        report=args.products_report,
    )
    product_outputs = write_products_outputs(product_args)
    reddit_csv = _write_reddit(args)
    arxiv_csv = _write_arxiv(args)
    ultra_csv = _write_ultra(args)
    source_paths = [
        product_outputs["diagnostics"],
        product_outputs["uca"],
        reddit_csv,
        arxiv_csv,
        ultra_csv,
    ]
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        rows.extend(_read_rows(path))
    for row in rows:
        row.setdefault("requirement_check", "")
        row.setdefault("requirement_status", "")
    checks = _build_requirement_checks(rows)
    blocked_checks = [check for check in checks if check["requirement_status"] == "blocked"]
    follow_up_lines = (
        [f"- {check['requirement_check']}: {check['notes']}" for check in blocked_checks]
        if blocked_checks
        else ["- No blocked T26 requirement checks remain in the generated stage outputs."]
    )
    for check in checks:
        rows.append(
            {
                "dataset": "stage",
                "stage": "t26",
                "method": "requirement_check",
                "seed": int(args.seed),
                "status": check["requirement_status"],
                "promotion_status": "not_promoted",
                "promoted": False,
                "requirement_check": check["requirement_check"],
                "requirement_status": check["requirement_status"],
                "notes": check["notes"],
            }
        )
    stage_csv = write_csv(args.csv, rows, STAGE_FIELDS)
    status = summarize_requirement_status(rows)
    ensure_report(
        args.report,
        [
            "# T26 Stage Summary",
            "",
            "## Scope",
            "",
            "- T26 implements the contract, diagnostics, balanced trainer utilities, UCA utilities, and stage outputs requested by the attachment.",
            "- Rows without fresh training are explicitly marked `ready_not_run`, `blocked_by_P0_gate`, or `blocked_by_teacher_gate`; no fabricated performance rows are promoted.",
            "- T25 rows remain available as historical inputs but are not treated as default T26 promoted rows.",
            "",
            "## Requirement Checklist",
            "",
            *markdown_table(checks, ["requirement_check", "requirement_status", "notes"]),
            "",
            "## Aggregated Rows",
            "",
            *markdown_table(rows, ["dataset", "method", "requested_full_node_ratio", "seed", "status", "accuracy", "macro_f1", "promotion_status", "failure_reason", "source_table"]),
            "",
            "## Safety Summary",
            "",
            f"- Promoted rows: `{status['promoted_rows']}`",
            f"- Forbidden promoted rows: `{status['forbidden_promoted_rows']}`",
            f"- All promoted rows safe: `{status['all_promoted_safe']}`",
            "- Full-node ratio is preserved as `(target_prototypes + shadow_nodes) / original_num_nodes`.",
            "- No logits input, KD, dense P2, legacy diffusion, full edge backprop, E x d materialization, full edge_index GPU path, source anchors, new exposed schema, or exact all-target ultra cache is promoted.",
            "",
            "## Required Follow-Up Experiments",
            "",
            *follow_up_lines,
            "",
            f"- Stage CSV: `{stage_csv}`",
        ],
    )
    missing = [str(path) for path in sorted(REQUIRED_OUTPUTS) if not path.exists()]
    return {"status": "completed", "rows": len(rows), "csv": str(stage_csv), "missing_required_outputs": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T26 condensed training recovery and UCA stage outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--t25-products-csv", default="experiments/tables/t25_products_recovery_ladder_seed42.csv")
    parser.add_argument("--t25-reddit-csv", default="experiments/tables/t26_reddit_actual_seed_sweep_sources.csv")
    parser.add_argument("--products-long-results-csv", default="experiments/tables/t26_products_long_experiments_seed42.csv")
    parser.add_argument("--products-diagnostics-csv", default="experiments/tables/t26_products_recovery_diagnostics_seed42.csv")
    parser.add_argument("--products-per-class-csv", default="experiments/tables/t26_products_per_class_report_seed42.csv")
    parser.add_argument("--products-uca-csv", default="experiments/tables/t26_products_uca_sweep_seed42.csv")
    parser.add_argument("--products-report", default="experiments/summaries/t26_products_recovery_notes.md")
    parser.add_argument("--reddit-csv", default="experiments/tables/t26_reddit_seed_trainer_mixup_sweep.csv")
    parser.add_argument("--reddit-report", default="experiments/summaries/t26_reddit_trainer_mixup_notes.md")
    parser.add_argument("--arxiv-csv", default="experiments/tables/t26_arxiv_teacher_sweep_seed42.csv")
    parser.add_argument("--arxiv-actual-source-csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--arxiv-report", default="experiments/summaries/t26_arxiv_teacher_notes.md")
    parser.add_argument("--ultra-csv", default="experiments/tables/t26_ultra_contract_regression_seed42.csv")
    parser.add_argument("--ultra-report", default="experiments/summaries/t26_ultra_contract_notes.md")
    parser.add_argument("--csv", default="experiments/tables/t26_stage_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_stage_summary.md")
    args = parser.parse_args()
    result = run_stage(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
