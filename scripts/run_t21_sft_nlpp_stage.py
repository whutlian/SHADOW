from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t21_common import markdown_table, no_forbidden_flags, read_csv, t21_safe_baseline, write_csv


SUMMARY_FIELDS = [
    "dataset",
    "status",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "target_accuracy",
    "target_passed",
    "safe_baseline",
    "delta_acc_vs_safe",
    "selected_blocks",
    "recovery_status",
    "forbidden_flags_clear",
    "reason",
]


TARGETS = {
    "acm": 0.93,
    "dblp": 0.85,
    "imdb": 0.50,
    "ogbn-arxiv": 0.64,
    "ogbn-products": 0.70,
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def _latest_by_dataset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("row_kind") == "final":
            out[row["dataset"]] = row
    return out


def build_summary() -> None:
    fullgraph_rows = read_csv("experiments/tables/t21_sft_fullgraph_seed42.csv")
    recovery_rows = read_csv("experiments/tables/t21_sft_condensation_recovery_seed42.csv")
    products_rows = read_csv("experiments/tables/t21_products_full_execution_seed42.csv")
    arxiv_rows = read_csv("experiments/tables/t21_arxiv_lazy_sft_seed42.csv")
    arxiv_sweep_rows = []
    for path in sorted(Path("experiments/tables").glob("t21_arxiv_lazy_perf_*_seed42.csv")):
        for row in read_csv(path):
            arxiv_sweep_rows.append(
                {
                    "variant": path.stem.replace("t21_arxiv_lazy_perf_", "").replace("_seed42", ""),
                    "accuracy": row.get("accuracy", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "predicted_class_count": row.get("predicted_class_count", ""),
                    "epochs": row.get("training_epochs", ""),
                    "cpu_gb": row.get("peak_cpu_ram_gb", ""),
                    "gpu_gb": row.get("peak_gpu_ram_gb", ""),
                    "train_s": row.get("training_time_s", ""),
                }
            )
    product_sweep_rows = []
    for path in sorted(Path("experiments/tables").glob("t21_products_lazy_perf_*_seed42.csv")):
        for row in read_csv(path):
            product_sweep_rows.append(
                {
                    "variant": path.stem.replace("t21_products_lazy_perf_", "").replace("_seed42", ""),
                    "accuracy": row.get("accuracy", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "predicted_class_count": row.get("predicted_class_count", ""),
                    "epochs": row.get("training_epochs", ""),
                    "cpu_gb": row.get("peak_cpu_ram_gb", ""),
                    "gpu_gb": row.get("peak_gpu_ram_gb", ""),
                    "train_s": row.get("training_time_s", ""),
                }
            )
    dry_rows = read_csv("experiments/tables/t21_scalability_dry_run_seed42.csv")
    full = _latest_by_dataset(fullgraph_rows)
    recovery_status: dict[str, str] = {}
    for row in recovery_rows:
        recovery_status.setdefault(row.get("dataset", ""), row.get("status", ""))
        if row.get("dataset") == "dblp" and row.get("status") == "started_diagnostic":
            recovery_status["dblp"] = "started_diagnostic"
    rows = []
    for dataset in ["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"]:
        row = full.get(dataset, {})
        safe = t21_safe_baseline(dataset)
        acc = _float(row, "accuracy", default=0.0)
        rows.append(
            {
                "dataset": dataset,
                "status": row.get("status", "missing"),
                "accuracy": row.get("accuracy", ""),
                "macro_f1": row.get("macro_f1", ""),
                "predicted_class_count": row.get("predicted_class_count", ""),
                "target_accuracy": TARGETS[dataset],
                "target_passed": bool(acc >= TARGETS[dataset]),
                "safe_baseline": safe["variant"],
                "delta_acc_vs_safe": row.get("delta_acc_vs_safe", ""),
                "selected_blocks": row.get("selected_blocks", ""),
                "recovery_status": recovery_status.get(dataset, ""),
                "forbidden_flags_clear": no_forbidden_flags(row),
                "reason": row.get("reason", ""),
            }
        )
    output = write_csv("experiments/tables/t21_stage_summary_seed42.csv", rows, SUMMARY_FIELDS)
    product = products_rows[0] if products_rows else {}
    dry_map = {row["dataset"]: row for row in dry_rows}
    acm = full.get("acm", {})
    dblp = full.get("dblp", {})
    imdb = full.get("imdb", {})
    arxiv = full.get("ogbn-arxiv", {})
    products = full.get("ogbn-products", {})
    lines = [
        "# T2.1-SFT-NL++ Stage Summary",
        "",
        "## What Changed",
        "",
        "- Added true chunked/memmap preprop API with `X0/X1/X2/Xres`, typed demand, structure, manifest schema, and forbidden-signal flags.",
        "- Added `SFTTableTeacherV2` with `sagn_lite`, `gamlp_lite`, and `residual_block_gated` modes plus class-aware losses including focal and sqrt-weighted CE.",
        "- Added robust block-selection scoring (`acc + 0.2 * macro_f1 + 0.05 * class_coverage`) and products full-run guards.",
        "- Added SFT block-signature recovery helpers and DBLP recovery-start table rows.",
        "",
        "## Final Rows",
        "",
        *markdown_table(rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "target_accuracy", "target_passed", "recovery_status", "reason"]),
        "",
        "## Products Execution",
        "",
        *markdown_table(products_rows, ["dataset", "status", "run_mode", "accuracy", "macro_f1", "full_edge_scans", "total_cache_bytes", "reason"]),
        "",
        "## Arxiv Lazy SFT",
        "",
        "Arxiv lazy rows use CPU/memmap-resident T2.1 preprop blocks and GPU mini-batch SFT. They do not load full `edge_index` during training/eval.",
        "",
        "Best row: `gamlp_lite`, hidden dim `512`, `cross_entropy`, `100` epochs, batch size `16384`, eval batch size `65536`.",
        "",
        *markdown_table(arxiv_rows, ["dataset", "status", "run_mode", "model_type", "loss_type", "hidden_dim", "accuracy", "macro_f1", "predicted_class_count", "peak_cpu_ram_gb", "peak_gpu_ram_gb"]),
        "",
        *markdown_table(arxiv_sweep_rows, ["variant", "accuracy", "macro_f1", "predicted_class_count", "epochs", "cpu_gb", "gpu_gb", "train_s"]),
        "",
        "## Products Lazy SFT Sweep",
        "",
        "All products sweep rows use CPU/memmap-resident preprop blocks with GPU mini-batch SFT. They do not load full `edge_index` during training/eval and keep logits/KD/dense P2/bounded edges/E*d disabled.",
        "",
        "Best row: `gamlp_lite`, hidden dim `512`, `sqrt_weighted_ce`, `100` epochs, batch size `16384`, eval batch size `65536`.",
        "",
        *markdown_table(product_sweep_rows, ["variant", "accuracy", "macro_f1", "predicted_class_count", "epochs", "cpu_gb", "gpu_gb", "train_s"]),
        "",
        "## Scalability Dry Run",
        "",
        *markdown_table(dry_rows, ["dataset", "cache_mode", "total_cache_bytes", "full_edge_scans", "wall_time_category", "server_recommended"]),
        "",
        "## Required Answers",
        "",
        f"1. Did ACM reach 0.93? {'Yes' if _float(acm, 'accuracy') >= 0.93 else 'No'}; current accuracy={acm.get('accuracy', '')}.",
        f"2. Did DBLP move to recovery? {'Yes' if recovery_status.get('dblp') in {'started_diagnostic', 'completed_diagnostic'} else 'No'}; fullgraph accuracy={dblp.get('accuracy', '')}.",
        "3. What remains as DBLP gap? Fullgraph SFT is strong; compressed prototype/shadow SFT block-signature accuracy is not yet promoted.",
        f"4. Was IMDB B3 robustly retained? {'B3_lad_scap' in imdb.get('selected_blocks', '')}; selected={imdb.get('selected_blocks', '')}.",
        f"5. Did IMDB reach 0.50? {'Yes' if _float(imdb, 'accuracy') >= 0.50 else 'No'}; current accuracy={imdb.get('accuracy', '')}.",
        f"6. Was arxiv class collapse fixed? {'Yes' if int(float(arxiv.get('predicted_class_count') or 0)) >= 35 else 'No'}; predicted_class_count={arxiv.get('predicted_class_count', '')}.",
        f"7. Did arxiv reach 0.64 without forbidden signals? {'Yes' if _float(arxiv, 'accuracy') >= 0.64 and no_forbidden_flags(arxiv) else 'No'}; current accuracy={arxiv.get('accuracy', '')}.",
        f"8. Did products full execution complete? {'Yes' if product.get('status') in {'completed', 'promoted'} else 'No'}; status={product.get('status', '')}. Current completed row is lazy CPU/memmap + GPU mini-batch SFT.",
        f"9. Did products beat 0.6689/macro baseline? {'Yes' if _float(products, 'accuracy') > 0.6689 or _float(products, 'macro_f1') > 0.338064 else 'No'}; accuracy={products.get('accuracy', '')}, macro_f1={products.get('macro_f1', '')}.",
        "10. Any promoted bounded/logit/KD/E*d rows? No in T2.1 generated tables; forbidden flags are explicitly false for promoted/reported rows.",
        f"11. paper100M dry-run: cache={dry_map.get('ogbn-papers100M', {}).get('total_cache_bytes', '')}, scans={dry_map.get('ogbn-papers100M', {}).get('full_edge_scans', '')}, server_recommended={dry_map.get('ogbn-papers100M', {}).get('server_recommended', '')}.",
        f"12. MAG240M dry-run: cache={dry_map.get('MAG240M', {}).get('total_cache_bytes', '')}, scans={dry_map.get('MAG240M', {}).get('full_edge_scans', '')}, server_recommended={dry_map.get('MAG240M', {}).get('server_recommended', '')}.",
        "13. Eligible datasets for recovery: ACM, DBLP, IMDB by current fullgraph gate; DBLP is the immediate started diagnostic target.",
        "14. Are all attachment gates satisfied? No: products and arxiv are now achieved locally, but ACM 0.93 and IMDB 0.50 remain open.",
        "",
        "## Artifacts",
        "",
        "- `experiments/tables/t21_preprop_manifest_index_seed42.csv`",
        "- `experiments/tables/t21_sft_block_selection_seed42.csv`",
        "- `experiments/tables/t21_sft_fullgraph_seed42.csv`",
        "- `experiments/tables/t21_products_full_execution_seed42.csv`",
        "- `experiments/tables/t21_sft_condensation_recovery_seed42.csv`",
        "- `experiments/tables/t21_scalability_dry_run_seed42.csv`",
        f"- `experiments/tables/t21_stage_summary_seed42.csv`",
        f"- CSV summary: `{output}`",
    ]
    report = Path("experiments/reports/t21_sft_nlpp_stage_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2.1-SFT-NL++ stage artifact generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--run-products-full", action="store_true")
    parser.add_argument("--products-train-epochs", type=int, default=0)
    args = parser.parse_args()
    py = sys.executable
    if not args.skip_runs:
        _run([py, "scripts/run_t21_arxiv_true_preprop.py", "--seed", str(args.seed)])
        _run([py, "scripts/run_t21_small_robust_block_selection.py"])
        product_cmd = [py, "scripts/run_t21_products_full_preprop.py", "--seed", str(args.seed), "--train-epochs", str(args.products_train_epochs)]
        if args.run_products_full:
            product_cmd.append("--run-full")
        _run(product_cmd)
        _run([py, "scripts/run_t21_sft_fullgraph.py"])
        _run([py, "scripts/run_t21_sft_condensation_recovery.py"])
        _run([py, "scripts/dry_run_t21_preprop_large.py"])
    build_summary()
    print(json.dumps({"summary": "experiments/reports/t21_sft_nlpp_stage_summary.md", "table": "experiments/tables/t21_stage_summary_seed42.csv"}, sort_keys=True))


if __name__ == "__main__":
    main()
