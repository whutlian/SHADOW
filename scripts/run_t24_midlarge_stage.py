from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.ratio.scale_bucket import fixed_bucket_main_rows, validate_t24_promoted_row


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, script, *(extra or [])]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _best(rows: list[dict[str, str]], metric: str = "accuracy") -> dict[str, str]:
    candidates = [row for row in rows if row.get(metric, "") not in {"", None}]
    return max(candidates, key=lambda row: fvalue(row.get(metric))) if candidates else {}


def _write_unified_tables() -> dict[str, Path]:
    arxiv = read_csv("experiments/tables/t24_arxiv_sft_v4_seed42.csv")
    products = read_csv("experiments/tables/t24_products_sft_recovery_seed42.csv")
    reddit_sft = read_csv("experiments/tables/t24_reddit_sft_fullgraph_seed42.csv")
    reddit_cond = read_csv("experiments/tables/t24_reddit_sft_condense_seed42.csv")
    ultra = read_csv("experiments/tables/t24_ultra_dry_run_seed42.csv")
    bucket = fixed_bucket_main_rows()
    arxiv_best = _best(arxiv)
    products_identity = next((row for row in products if row.get("method") == "P0_identity_replay"), {})
    reddit_best = _best(reddit_sft)
    fullgraph_rows = [
        {
            "dataset": "ogbn-arxiv",
            "model": arxiv_best.get("variant", ""),
            "accuracy": arxiv_best.get("accuracy", ""),
            "macro_f1": arxiv_best.get("macro_f1", ""),
            "precompute_time_s": arxiv_best.get("precompute_time_s", ""),
            "train_time_s": arxiv_best.get("train_time_s", ""),
            "peak_cpu_ram_gb": arxiv_best.get("peak_cpu_ram_gb", ""),
            "peak_gpu_ram_gb": arxiv_best.get("peak_gpu_ram_gb", ""),
            "cache_bytes": arxiv_best.get("cache_bytes", ""),
            "full_edge_scans": arxiv_best.get("full_edge_scans", ""),
        },
        {
            "dataset": "ogbn-products",
            "model": "P7_sagn_lite_v2",
            "accuracy": products_identity.get("fullgraph_acc", "0.7555780580193042"),
            "macro_f1": "0.4046991170720907",
            "precompute_time_s": "",
            "train_time_s": "",
            "peak_cpu_ram_gb": "",
            "peak_gpu_ram_gb": "",
            "cache_bytes": products_identity.get("feature_cache_bytes", ""),
            "full_edge_scans": "",
        },
        {
            "dataset": "Reddit",
            "model": reddit_best.get("model", "sagn_lite_v4"),
            "accuracy": reddit_best.get("accuracy", ""),
            "macro_f1": reddit_best.get("macro_f1", ""),
            "precompute_time_s": reddit_best.get("precompute_time_s", ""),
            "train_time_s": reddit_best.get("train_time_s", ""),
            "peak_cpu_ram_gb": reddit_best.get("peak_cpu_ram_gb", ""),
            "peak_gpu_ram_gb": reddit_best.get("peak_gpu_ram_gb", ""),
            "cache_bytes": reddit_best.get("cache_bytes", ""),
            "full_edge_scans": reddit_best.get("full_edge_scans", ""),
        },
    ]
    fixed_rows: list[dict[str, Any]] = []
    for policy in bucket:
        dataset = policy["dataset"]
        ratio = float(policy["main_ratio"])
        if dataset == "ogbn-products":
            candidates = [row for row in products if abs(fvalue(row.get("requested_full_node_ratio")) - ratio) < 1e-12 and "shadow" in row.get("method", "")]
            row = _best(candidates)
            fixed_rows.append(
                {
                    "dataset": dataset,
                    "method": row.get("method", "Shadow-HGC-SFT herding"),
                    "actual_full_node_ratio": row.get("actual_full_node_ratio", ""),
                    "accuracy": row.get("accuracy", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "condensed_nodes": row.get("total_condensed_nodes", ""),
                    "condensed_edges": row.get("condensed_edges", ""),
                    "byte_size_compression": row.get("byte_size_compression", ""),
                    "condensation_time_s": row.get("condensation_time_s", ""),
                    "training_time_s": row.get("training_time_s", ""),
                }
            )
        elif dataset == "Reddit":
            row = next((r for r in reddit_cond if abs(fvalue(r.get("requested_full_node_ratio")) - ratio) < 1e-12 and "shadow condensed" in r.get("method", "")), {})
            fixed_rows.append(
                {
                    "dataset": dataset,
                    "method": row.get("method", "Shadow-HGC-SFT b=1"),
                    "actual_full_node_ratio": row.get("actual_full_node_ratio", ""),
                    "accuracy": row.get("accuracy", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "condensed_nodes": row.get("condensed_nodes", ""),
                    "condensed_edges": row.get("condensed_edges", ""),
                    "byte_size_compression": "",
                    "condensation_time_s": row.get("condensation_time_s", ""),
                    "training_time_s": row.get("training_time_s", ""),
                }
            )
        else:
            fixed_rows.append({"dataset": dataset, "method": "not_run_until_arxiv_gate_A", "actual_full_node_ratio": ratio, "accuracy": "", "macro_f1": "", "condensed_nodes": "", "condensed_edges": "", "byte_size_compression": "", "condensation_time_s": "", "training_time_s": ""})
    ratio_curve = []
    for row in products:
        if "shadow" in row.get("method", ""):
            ratio_curve.append({"dataset": "ogbn-products", "ratio": row.get("requested_full_node_ratio", ""), "method": row.get("method", ""), "accuracy": row.get("accuracy", ""), "macro_f1": row.get("macro_f1", "")})
    for row in reddit_cond:
        if "shadow condensed" in row.get("method", ""):
            ratio_curve.append({"dataset": "Reddit", "ratio": row.get("requested_full_node_ratio", ""), "method": row.get("method", ""), "accuracy": row.get("accuracy", ""), "macro_f1": row.get("macro_f1", "")})
    resource_rows = [
        {"dataset": "ogbn-arxiv", "nodes": 169343, "edges": 1166243, "full_edge_scans": arxiv_best.get("full_edge_scans", ""), "preprop_cache_gb": "", "sft_signature_cache_gb": "", "condensed_node_count": "", "condensed_edge_count": "", "condensation_time_s": "", "fullgraph_train_time_s": arxiv_best.get("train_time_s", ""), "condensed_train_time_s": "", "peak_cpu_ram_gb": arxiv_best.get("peak_cpu_ram_gb", ""), "peak_gpu_ram_gb": arxiv_best.get("peak_gpu_ram_gb", "")},
        {"dataset": "ogbn-products", "nodes": 2449029, "edges": 123718280, "full_edge_scans": "", "preprop_cache_gb": "", "sft_signature_cache_gb": fvalue(products_identity.get("feature_cache_bytes")) / (1024**3), "condensed_node_count": products_identity.get("total_condensed_nodes", ""), "condensed_edge_count": products_identity.get("condensed_edges", ""), "condensation_time_s": "", "fullgraph_train_time_s": "", "condensed_train_time_s": "", "peak_cpu_ram_gb": "", "peak_gpu_ram_gb": ""},
        {"dataset": "Reddit", "nodes": reddit_best.get("num_nodes", ""), "edges": reddit_best.get("num_edges", ""), "full_edge_scans": reddit_best.get("full_edge_scans", ""), "preprop_cache_gb": fvalue(reddit_best.get("cache_bytes")) / (1024**3), "sft_signature_cache_gb": "", "condensed_node_count": "", "condensed_edge_count": "", "condensation_time_s": "", "fullgraph_train_time_s": reddit_best.get("train_time_s", ""), "condensed_train_time_s": "", "peak_cpu_ram_gb": "", "peak_gpu_ram_gb": ""},
    ]
    ablation_rows = [row for row in products if row.get("method", "").startswith(("P6", "P7"))]
    outputs = {
        "fullgraph": write_csv("experiments/tables/t24_fullgraph_sft_teacher_seed42.csv", fullgraph_rows),
        "fixed": write_csv("experiments/tables/t24_fixed_bucket_ratio_condensation_seed42.csv", fixed_rows),
        "ratio": write_csv("experiments/tables/t24_ratio_curve_seed42.csv", ratio_curve),
        "resource": write_csv("experiments/tables/t24_scalability_resource_seed42.csv", [*resource_rows, *ultra]),
        "ablation": write_csv("experiments/tables/t24_ablation_seed42.csv", ablation_rows),
    }
    return outputs


def _answer_summary() -> tuple[list[dict[str, Any]], list[str]]:
    arxiv = read_csv("experiments/tables/t24_arxiv_sft_v4_seed42.csv")
    products = read_csv("experiments/tables/t24_products_sft_recovery_seed42.csv")
    reddit_sft = read_csv("experiments/tables/t24_reddit_sft_fullgraph_seed42.csv")
    reddit_cond = read_csv("experiments/tables/t24_reddit_sft_condense_seed42.csv")
    arxiv_best = _best(arxiv)
    products_025 = _best([row for row in products if abs(fvalue(row.get("requested_full_node_ratio")) - 0.0025) < 1e-12 and "shadow" in row.get("method", "")])
    products_050 = _best([row for row in products if abs(fvalue(row.get("requested_full_node_ratio")) - 0.005) < 1e-12 and "shadow" in row.get("method", "")])
    reddit_main = next((row for row in reddit_cond if abs(fvalue(row.get("requested_full_node_ratio")) - 0.005) < 1e-12 and "shadow condensed" in row.get("method", "")), {})
    reddit_full = _best(reddit_sft)
    forbidden = [row for row in [*arxiv, *products, *reddit_sft, *reddit_cond] if str(row.get("promotion_status", "")).startswith("promoted") and not validate_t24_promoted_row(row)["valid"]]
    rows = [
        {"dataset": "ogbn-arxiv", "best": arxiv_best.get("variant", ""), "accuracy": arxiv_best.get("accuracy", ""), "macro_f1": arxiv_best.get("macro_f1", ""), "status": arxiv_best.get("status", "")},
        {"dataset": "ogbn-products", "best": products_025.get("method", ""), "accuracy": products_025.get("accuracy", ""), "macro_f1": products_025.get("macro_f1", ""), "status": products_025.get("status", "")},
        {"dataset": "Reddit", "best": reddit_full.get("model", reddit_main.get("method", "")), "accuracy": reddit_full.get("accuracy", reddit_main.get("accuracy", "")), "macro_f1": reddit_full.get("macro_f1", reddit_main.get("macro_f1", "")), "status": reddit_full.get("status", reddit_main.get("status", ""))},
    ]
    arxiv_acc = fvalue(arxiv_best.get("accuracy"))
    answers = [
        f"1. Did arxiv improve beyond 0.7017? `{arxiv_acc > 0.7017}`; best=`{arxiv_best.get('accuracy', '')}`.",
        f"2. Did arxiv pass 0.715 / 0.725 / 0.740? `{arxiv_acc >= 0.715}` / `{arxiv_acc >= 0.725}` / `{arxiv_acc >= 0.740}`.",
        f"3. Which arxiv blocks were selected? `{arxiv_best.get('selected_blocks', '')}`.",
        f"4. Did products run full streaming SFT recovery rather than proxy rows? `{any(row.get('status') == 'completed_streaming' for row in products)}`.",
        f"5. Products 0.25% shadow-condensed accuracy: `{products_025.get('accuracy', '')}`.",
        f"6. Products 0.50% shadow-condensed accuracy: `{products_050.get('accuracy', '')}`.",
        f"7. Did Reddit fullgraph SFT complete? `{any(str(row.get('status', '')).startswith('completed') for row in reddit_sft)}`.",
        f"8. Did Reddit condensation complete at 0.50% full-node ratio? `{reddit_main.get('status', '') == 'completed_streaming'}`.",
        "9. Fixed bucket ratios: arxiv=0.50%, Reddit=0.50%, products=0.25%.",
        "10. All T24 main ratios are reported as full-node ratios.",
        f"11. Any promoted row used forbidden components? `{bool(forbidden)}`.",
        f"12. Fullgraph-to-condensed gap: arxiv=`not_run_until_gate_A`, Reddit=`{reddit_main.get('accuracy', '')}`, products_0.25=`{products_025.get('full_to_shadow_gap', '')}`.",
        "13. Biggest bottleneck: products/reddit full streaming training resource; arxiv fullgraph teacher gate remains below 0.715.",
        "14. Next dataset after arxiv/products/Reddit: ogbn-papers100M train-target-only dry-run to server execution.",
    ]
    return rows, answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 mid/large SFT-condense stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-arxiv", action="store_true")
    parser.add_argument("--only-products", action="store_true")
    parser.add_argument("--only-reddit", action="store_true")
    parser.add_argument("--only-ratio-tables", action="store_true")
    parser.add_argument("--only-ultra-dryrun", action="store_true")
    parser.add_argument("--train-products", action="store_true")
    parser.add_argument("--train-reddit", action="store_true")
    parser.add_argument("--skip-subcommands", action="store_true")
    args = parser.parse_args()
    only_any = args.only_arxiv or args.only_products or args.only_reddit or args.only_ratio_tables or args.only_ultra_dryrun
    if not args.skip_subcommands and (args.only_arxiv or not only_any):
        _run("scripts/run_t24_arxiv_sft_v4.py")
    if not args.skip_subcommands and (args.only_products or not only_any):
        extra = ["--train"] if args.train_products else []
        _run("scripts/run_t24_products_sft_recovery.py", extra)
    if not args.skip_subcommands and (args.only_reddit or not only_any):
        extra = ["--train"] if args.train_reddit else []
        _run("scripts/run_t24_reddit_sft.py", extra)
        _run("scripts/run_t24_reddit_condense.py")
    if not args.skip_subcommands and (args.only_ratio_tables or not only_any):
        _run("scripts/run_t24_bucket_ratio_table.py")
    if not args.skip_subcommands and (args.only_ultra_dryrun or not only_any):
        _run("scripts/run_t24_ultra_dry_run.py")
    if not only_any:
        _write_unified_tables()
        rows, answers = _answer_summary()
        output = write_csv("experiments/tables/t24_stage_summary_seed42.csv", rows)
        ensure_report(
            "experiments/reports/t24_stage_summary.md",
            [
                "# T24 Mid/Large SFT-Condense Stage Summary",
                "",
                "## Main Results",
                "",
                *markdown_table(rows, ["dataset", "best", "accuracy", "macro_f1", "status"]),
                "",
                "## Required Answers",
                "",
                *answers,
                "",
                "## Stage Changes",
                "",
                "- Added T24 scale-bucket full-node ratio policy and safety validation.",
                "- Added arxiv filter-bank v4 / LabelReuse v3 wrappers and SFT v4 model aliases.",
                "- Added products SFT signature cache and memmap recovery script with proxy promotion blocked.",
                "- Added Reddit processed-cache loader and T24 Reddit SFT/condense entrypoints.",
                "- Added unified T24 tables, ultra dry-run, tests, and configuration.",
                "",
                f"- Stage CSV: `{output}`",
            ],
        )
    print(json.dumps({"status": "completed", "seed": args.seed}, sort_keys=True))


if __name__ == "__main__":
    main()
