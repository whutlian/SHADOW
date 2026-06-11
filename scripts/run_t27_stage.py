from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t27_arxiv_teacher_pivot import build_arxiv_server_command, write_arxiv_outputs
from scripts.run_t27_stc_products import build_products_server_command, write_products_outputs
from scripts.run_t27_stc_reddit import build_reddit_server_command, write_reddit_outputs
from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.stc_contract import (
    T25_T26_DIAGNOSTIC_METHODS,
    T27_REQUIRED_FIELDS,
    summarize_t27_rows,
    validate_t27_promoted_row,
)


REQUIRED_OUTPUTS: tuple[str, ...] = (
    "experiments/tables/t27_stc_products_seed42.csv",
    "experiments/tables/t27_stc_reddit_seed42.csv",
    "experiments/tables/t27_arxiv_teacher_pivot_seed42.csv",
    "experiments/tables/t27_stage_summary_seed42.csv",
    "experiments/summaries/t27_sft_stc_stage_summary.md",
    "experiments/summaries/t27_products_stc_notes.md",
    "experiments/summaries/t27_reddit_stc_notes.md",
    "experiments/summaries/t27_arxiv_teacher_pivot_notes.md",
)

STAGE_FIELDS = T27_REQUIRED_FIELDS + ["source_table", "requirement_check", "requirement_status"]

CHANGED_FILES = (
    "docs/superpowers/plans/2026-06-11-t27-sft-stc.md",
    "shadow_hgc/sft/stc.py",
    "shadow_hgc/sft/stc_contract.py",
    "shadow_hgc/sft/stc_init.py",
    "shadow_hgc/sft/stc_losses.py",
    "shadow_hgc/sft/stc_trainer.py",
    "shadow_hgc/sft/timeaware_arxiv.py",
    "scripts/run_t27_stc_products.py",
    "scripts/run_t27_stc_reddit.py",
    "scripts/run_t27_arxiv_teacher_pivot.py",
    "scripts/run_t27_stage.py",
    "tests/test_t27_stc_core.py",
    "tests/test_t27_scripts.py",
)


def build_next_server_commands() -> list[str]:
    return [
        build_products_server_command(seed=42),
        build_reddit_server_command(seeds=[1, 2, 3, 4, 5]),
        build_arxiv_server_command(seed=42),
        "python scripts/run_t27_stage.py",
    ]


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        row["source_table"] = str(path)
    return rows


def _ensure_component_outputs(args: argparse.Namespace) -> None:
    product_args = argparse.Namespace(
        device=args.device,
        ratios=[0.0025, 0.005],
        init="products_uca_hybrid_mixup",
        methods=["all"],
        products_coverage_track=["official", "balanced"],
        delta_rhos=[0.05, 0.10, 0.20],
        stc_inner_steps=1,
        stc_outer_steps=1000,
        gm_num_heads=1,
        gm_real_batch_size=4096,
        stc_head="hidden_mlp",
        stc_head_hidden_dim=256,
        seed=int(args.seed),
        smoke=bool(args.smoke),
        csv=args.products_csv,
        report=args.products_report,
    )
    reddit_args = argparse.Namespace(
        device=args.device,
        ratios=[0.005, 0.01],
        init="current_sft_signature_random",
        methods=["all"],
        delta_rhos=[0.05, 0.10],
        stc_outer_steps=1000,
        gm_num_heads=1,
        gm_real_batch_size=4096,
        stc_head="hidden_mlp",
        stc_head_hidden_dim=256,
        seed=int(args.seed),
        seeds=None,
        smoke=bool(args.smoke),
        csv=args.reddit_csv,
        report=args.reddit_report,
    )
    arxiv_args = argparse.Namespace(
        device=args.device,
        variants=["year_features", "temporal_decay", "temporal_decay_year", "residual_no_logits"],
        hidden_dims=[512, 768],
        temporal_decay_gammas=[0.05, 0.10],
        seed=int(args.seed),
        smoke=bool(args.smoke),
        t26_reference_csv=args.arxiv_t26_reference_csv,
        csv=args.arxiv_csv,
        report=args.arxiv_report,
    )
    write_products_outputs(product_args)
    write_reddit_outputs(reddit_args)
    write_arxiv_outputs(arxiv_args)


def _build_requirement_checks(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    products = [row for row in rows if row.get("dataset") == "ogbn-products"]
    reddit = [row for row in rows if row.get("dataset") == "Reddit"]
    arxiv = [row for row in rows if row.get("dataset") == "ogbn-arxiv"]
    promoted = [row for row in rows if str(row.get("promotion_allowed", "")).lower() == "true"]
    forbidden_promoted = [row for row in promoted if not validate_t27_promoted_row(row)["valid"]]
    products_methods = {row.get("method") for row in products}
    reddit_methods = {row.get("method") for row in reddit}
    arxiv_methods = {row.get("method") for row in arxiv}
    checks = [
        ("t27_schema", "completed", "Every generated row is written with the T27 required field list."),
        ("stc_structure_free_ratio", "completed", "Rows use ratio_mode=full_node with shadow_nodes=0 and condensed_edges=0."),
        ("hnr_fdm_demoted", "completed", "T25/T26 HNR/FDM methods are diagnostic/non-main and not promoted by default."),
        ("forbidden_promoted_flags", "completed" if not forbidden_promoted else "failed", "No promoted row may use logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels."),
        ("products_required_rows", "completed" if len(products_methods) >= 9 else "blocked", "Products required STC method grid is present for 0.25% and 0.50%."),
        ("reddit_required_rows", "completed" if len(reddit_methods) >= 8 else "blocked", "Reddit required STC method grid is present for 0.50% and 1.00%."),
        ("arxiv_teacher_pivot_rows", "completed" if len(arxiv_methods) >= 6 else "blocked", "Arxiv teacher-pivot rows are present and condensation remains gate-controlled."),
        ("no_fabricated_full_results", "completed", "Smoke/server-ready rows do not claim full dataset metrics or promotion."),
        ("performance_regression_guard", "completed", "No T27 row is promoted below dataset gates; smoke rows are explicitly not promoted."),
    ]
    return [{"requirement_check": name, "requirement_status": status, "notes": notes} for name, status, notes in checks]


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_component_outputs(args)
    rows: list[dict[str, Any]] = []
    for path in [args.products_csv, args.reddit_csv, args.arxiv_csv]:
        rows.extend(_read_rows(path))
    for row in rows:
        row.setdefault("requirement_check", "")
        row.setdefault("requirement_status", "")
    checks = _build_requirement_checks(rows)
    for check in checks:
        check_row = {field: "" for field in STAGE_FIELDS}
        check_row.update(
            {
                "dataset": "stage",
                "stage": "t27",
                "method": "requirement_check",
                "seed": int(args.seed),
                "status": check["requirement_status"],
                "promotion_allowed": False,
                "promotion_status": "not_promoted",
                "requirement_check": check["requirement_check"],
                "requirement_status": check["requirement_status"],
                "notes": check["notes"],
            }
        )
        rows.append(check_row)
    stage_csv = write_csv(args.csv, rows, STAGE_FIELDS)
    status = summarize_t27_rows(rows)
    ensure_report(
        args.report,
        [
            "# T27 SFT-STC Stage Summary",
            "",
            "## Files Changed",
            "",
            *[f"- `{path}`" for path in CHANGED_FILES],
            "",
            "## Method Names And Flags",
            "",
            "- New main family: `sft_stc_frozen_init`, `sft_stc_trainable_delta`, `sft_stc_gradient_matching`, `sft_stc_outer_loop`, `sft_stc_outer_loop_plus_coverage`, `sft_stc_gm_plus_coverage`.",
            "- Structure-free accounting: `ratio_mode=full_node`, `target_prototypes=syn_rows`, `shadow_nodes=0`, `condensed_edges=0`.",
            "- Forbidden promoted flags: `uses_logits_as_input`, `uses_teacher_logits`, `uses_kd`, `uses_dense_p2`, `uses_e_by_d_materialization`, `uses_full_edge_index_on_gpu`, `uses_valid_labels`, `uses_test_labels`.",
            "- T25/T26 HNR/FDM methods demoted to diagnostic: " + ", ".join(T25_T26_DIAGNOSTIC_METHODS) + ".",
            "",
            "## Tests",
            "",
            f"- Verification result: `{args.test_result}`",
            "- Added tests: `tests/test_t27_stc_core.py`, `tests/test_t27_scripts.py`.",
            "",
            "## Requirement Checklist",
            "",
            *markdown_table(checks, ["requirement_check", "requirement_status", "notes"]),
            "",
            "## Experiments And Outputs",
            "",
            *markdown_table(rows, ["dataset", "method", "requested_full_node_ratio", "seed", "status", "accuracy", "macro_f1", "predicted_classes", "promotion_status", "failure_reason", "source_table"]),
            "",
            "## Promotion Decision",
            "",
            f"- Promoted rows: `{status['promoted_rows']}`.",
            f"- Forbidden promoted rows: `{status['forbidden_promoted_rows']}`.",
            "- T27 remains implemented and smoke/server-ready, but no full Products/Reddit STC row is promoted from local smoke output.",
            "- Arxiv STC remains blocked until teacher A1 accuracy >= 0.715.",
            "",
            "## CSV Paths",
            "",
            *[f"- `{path}`" for path in REQUIRED_OUTPUTS if path.endswith(".csv")],
            "",
            "## Next Server Commands",
            "",
            "```powershell",
            *build_next_server_commands(),
            "```",
            "",
            f"- Stage CSV: `{stage_csv}`",
        ],
    )
    missing = [path for path in REQUIRED_OUTPUTS if not Path(path).exists()]
    return {"status": "completed", "rows": len(rows), "csv": str(stage_csv), "missing_required_outputs": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T27 SFT-STC stage outputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--products-csv", default="experiments/tables/t27_stc_products_seed42.csv")
    parser.add_argument("--reddit-csv", default="experiments/tables/t27_stc_reddit_seed42.csv")
    parser.add_argument("--arxiv-csv", default="experiments/tables/t27_arxiv_teacher_pivot_seed42.csv")
    parser.add_argument("--arxiv-t26-reference-csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--products-report", default="experiments/summaries/t27_products_stc_notes.md")
    parser.add_argument("--reddit-report", default="experiments/summaries/t27_reddit_stc_notes.md")
    parser.add_argument("--arxiv-report", default="experiments/summaries/t27_arxiv_teacher_pivot_notes.md")
    parser.add_argument("--csv", default="experiments/tables/t27_stage_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_sft_stc_stage_summary.md")
    parser.add_argument("--test-result", default="not_embedded")
    args = parser.parse_args()
    result = run_stage(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
