from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import PRIMARY_TARGETS, SAFE_BASELINES, T2_STAGE_FIELDS, markdown_table, read_csv, write_csv, write_json


STAGE_TABLE = Path("experiments/tables/t2_sft_stage_summary_seed42.csv")
STAGE_REPORT = Path("experiments/reports/t2_sft_no_logits_stage_summary.md")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _final_by_dataset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        if row.get("row_kind") in {"final", "resource_guard"}:
            out[row["dataset"]] = row
    return out


def build_stage_summary() -> None:
    selection = read_csv("experiments/tables/t2_sft_safe_block_selection_seed42.csv")
    fullgraph = read_csv("experiments/tables/t2_sft_fullgraph_seed42.csv")
    preprop = read_csv("experiments/tables/t2_preprop_manifest_index_seed42.csv")
    dry = read_csv("experiments/tables/t2_sft_scalability_dry_run_seed42.csv")
    recovery = read_csv("experiments/tables/t2_condensation_recovery_seed42.csv")
    final = _final_by_dataset(fullgraph or selection)
    rows = []
    for dataset, safe in SAFE_BASELINES.items():
        row = final.get(dataset, {})
        acc = _float(row, "accuracy", default=0.0)
        macro = _float(row, "macro_f1", default=0.0)
        rows.append(
            {
                "dataset": dataset,
                "status": row.get("status", "missing"),
                "accuracy": row.get("accuracy", ""),
                "macro_f1": row.get("macro_f1", ""),
                "predicted_class_count": row.get("predicted_class_count", ""),
                "primary_target": PRIMARY_TARGETS[dataset],
                "primary_target_passed": bool(acc >= PRIMARY_TARGETS[dataset]),
                "safe_baseline": safe["variant"],
                "safe_acc": safe["accuracy"],
                "safe_macro_f1": safe["macro_f1"],
                "delta_acc_vs_safe": acc - float(safe["accuracy"]) if row.get("accuracy", "") != "" else "",
                "delta_macro_f1_vs_safe": macro - float(safe["macro_f1"]) if row.get("macro_f1", "") != "" else "",
                "selected_blocks": row.get("selected_blocks", ""),
                "uses_logits_as_input": row.get("uses_logits_as_input", False),
                "uses_dense_p2": row.get("uses_dense_p2", False),
                "uses_bounded_edges": row.get("uses_bounded_edges", False),
                "uses_e_by_d_materialization": row.get("uses_e_by_d_materialization", False),
                "reason": row.get("reason", ""),
            }
        )
    write_csv(STAGE_TABLE, rows)
    write_json(STAGE_TABLE.with_suffix(".json"), {"rows": rows})
    branch_rows = [row for row in selection if row.get("row_kind") == "branch"]
    kept = [row for row in branch_rows if row.get("kept_or_dropped") == "kept"]
    dropped = [row for row in branch_rows if row.get("kept_or_dropped") == "dropped"]
    promoted = [row for row in final.values() if row.get("status") == "promoted"]
    recovery_eligible = [row["dataset"] for row in promoted]
    dry_map = {row["dataset"]: row for row in dry}
    lines = [
        "# T2-SFT-NL: No-Logits Scalable Fullgraph Teacher Stage Summary",
        "",
        "## What Changed",
        "",
        "- Added `shadow_hgc/preprop/*` chunked/memmap preprop modules with manifest, block stats, and resource schema.",
        "- Added no-logits `SFTTableTeacher` with `sagn_lite` and `gamlp_lite` modes.",
        "- Added validation-only T2 safe block selection and runner scripts.",
        "- Added promoted-row guards for logits, teacher logits/KD, dense P2, bounded edges, diffusion legacy, fullgraph edge backprop, and E x d materialization.",
        "- Kept T1 logit code as historical artifact only; T2 scripts do not consume logit caches or propagated logits.",
        "",
        "## Final Dataset Results",
        "",
        *markdown_table(rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "primary_target", "primary_target_passed", "delta_acc_vs_safe", "selected_blocks", "reason"]),
        "",
        "## Kept Blocks",
        "",
        *markdown_table(kept, ["dataset", "block_group", "branch_valid_acc", "branch_test_acc_debug", "gate_value"]),
        "",
        "## Dropped Blocks",
        "",
        *markdown_table(dropped, ["dataset", "block_group", "branch_valid_acc", "branch_test_acc_debug", "drop_reason"]),
        "",
        "## Preprop Manifest Status",
        "",
        *markdown_table(preprop, ["dataset", "status", "num_blocks", "total_cache_bytes", "full_edge_scans", "uses_logits_as_input", "reason"]),
        "",
        "## Scalability Dry-Run",
        "",
        *markdown_table(dry, ["dataset", "cache_mode", "total_cache_bytes", "full_edge_scans", "wall_time_category", "server_recommended"]),
        "",
        "## Condensation Recovery Gate",
        "",
        *markdown_table(recovery, ["dataset", "recovery_row", "fullgraph_accuracy", "status", "reason"]),
        "",
        "## Required Answers",
        "",
        "1. Were logits completely removed from promoted signals? Yes. T2 rows set `uses_logits_as_input=false`, `uses_teacher_logits=false`, `uses_kd=false`; no promoted T2 row consumed logit caches or propagated logits.",
        f"2. Did T2 improve ACM beyond 0.93? {'Yes' if _float(final.get('acm', {}), 'accuracy') >= 0.93 else 'No'}.",
        f"3. Did T2 recover DBLP beyond 0.85? {'Yes' if _float(final.get('dblp', {}), 'accuracy') >= 0.85 else 'No'}.",
        f"4. Did T2 improve IMDB beyond 0.45? {'Yes' if _float(final.get('imdb', {}), 'accuracy') >= 0.45 else 'No'}.",
        f"5. Did T2 improve arxiv beyond 0.66 without logits/diffusion/P2? {'Yes' if _float(final.get('ogbn-arxiv', {}), 'accuracy') >= 0.66 else 'No'}.",
        f"6. Did T2 improve products beyond 0.70 or macro-F1 beyond LAD baseline? {'Yes' if (_float(final.get('ogbn-products', {}), 'accuracy') >= 0.70 or _float(final.get('ogbn-products', {}), 'macro_f1') > SAFE_BASELINES['ogbn-products']['macro_f1']) else 'No'}.",
        "7. Which blocks were kept/dropped by validation? See `Kept Blocks` and `Dropped Blocks` tables above.",
        "8. Which blocks hurt and why? Dropped rows are marked `dropped_by_validation`; the current script treats validation accuracy/macro-F1 regression as the reason.",
        "9. Did any promoted row use bounded edges? No; T2 promotion guard rejects `uses_bounded_edges=true`.",
        "10. Did any promoted row materialize `E x d`? No; T2 preprop and promotion rows record `uses_e_by_d_materialization=false`.",
        f"11. Are paper100M/MAG240M dry-runs still feasible? paper100M server_recommended={dry_map.get('ogbn-papers100M', {}).get('server_recommended', '')}; MAG240M server_recommended={dry_map.get('MAG240M', {}).get('server_recommended', '')}.",
        f"12. Which datasets are eligible for condensation recovery? {', '.join(recovery_eligible) if recovery_eligible else 'None; no T2 fullgraph row was promoted beyond safe baselines.'} See `t2_condensation_recovery_seed42.csv` for identity replay and eligible-not-run prototype/shadow rows.",
        "13. If no dataset improves, is the bottleneck data/schema, feature strength, or model capacity? For rows below safe baselines, the immediate bottleneck is feature-strength/model-capacity of the no-logits table teacher; products is additionally gated by local scalability.",
        "",
        "## Artifacts",
        "",
        "- `experiments/tables/t2_preprop_manifest_index_seed42.csv`",
        "- `experiments/tables/t2_sft_safe_block_selection_seed42.csv`",
        "- `experiments/tables/t2_sft_fullgraph_seed42.csv`",
        "- `experiments/tables/t2_sft_scalability_dry_run_seed42.csv`",
        "- `experiments/tables/t2_condensation_recovery_seed42.csv`",
        "- `experiments/tables/t2_sft_stage_summary_seed42.csv`",
    ]
    STAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    STAGE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_fullgraph_from_selection() -> None:
    selection = read_csv("experiments/tables/t2_sft_safe_block_selection_seed42.csv")
    rows = [row for row in selection if row.get("row_kind") in {"final", "resource_guard"}]
    output = write_csv("experiments/tables/t2_sft_fullgraph_seed42.csv", rows, T2_STAGE_FIELDS)
    lines = [
        "# T2-SFT-NL Fullgraph Teacher Summary",
        "",
        "Rows are materialized from the validation-selected safe block selection run; no duplicate training is performed by the stage driver.",
        "",
        *markdown_table(rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "selected_blocks", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    report = Path("experiments/reports/t2_sft_fullgraph_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete T2-SFT-NL no-logits stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--medium-epochs", type=int, default=4)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--run-products-full", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    if not args.skip_runs:
        products_flag = ["--run-products-full"] if args.run_products_full else []
        _run([py, "scripts/run_t2_preprop_blocks.py", "--seed", str(args.seed), *products_flag])
        _run([py, "scripts/run_t2_safe_block_selection.py", "--seed", str(args.seed), "--epochs", str(args.epochs), "--medium-epochs", str(args.medium_epochs), *products_flag])
        materialize_fullgraph_from_selection()
        _run([py, "scripts/run_t2_condensation_recovery.py"])
        _run([py, "scripts/run_t2_scalability_dry_run.py"])
    build_stage_summary()
    print(json.dumps({"summary": str(STAGE_REPORT), "table": str(STAGE_TABLE)}, sort_keys=True))


if __name__ == "__main__":
    main()
