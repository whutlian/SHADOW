from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t2_safe_block_selection import run_dataset_selection
from scripts.t2_common import ALL_T2_DATASETS, T2_STAGE_FIELDS, markdown_table, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2 no-logits SFT fullgraph final rows.")
    parser.add_argument("--datasets", nargs="+", default=ALL_T2_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--medium-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--medium-hidden-dim", type=int, default=256)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--medium-block-dim", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--edge-limit", type=int, default=0)
    parser.add_argument("--medium-batch-size", type=int, default=16384)
    parser.add_argument("--scap-topk", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--selection-epsilon-acc", type=float, default=0.0)
    parser.add_argument("--selection-epsilon-f1", type=float, default=0.0)
    parser.add_argument("--run-products-full", action="store_true")
    parser.add_argument("--output", default="experiments/tables/t2_sft_fullgraph_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t2_sft_fullgraph_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t2_sft_fullgraph_seed42")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows.extend(row for row in run_dataset_selection(dataset, args) if row.get("row_kind") in {"final", "resource_guard"})
    output = write_csv(args.output, rows, T2_STAGE_FIELDS)
    write_json(Path(args.output).with_suffix(".json"), {"rows": rows})
    lines = [
        "# T2-SFT-NL Fullgraph Teacher Summary",
        "",
        "Rows are validation-selected SAGN/GAMLP-lite table teachers. Test metrics are report-only.",
        "",
        *markdown_table(rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "selected_blocks", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
