from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t25_contract import T25_OUTPUT_FIELDS


SUMMARY_FIELDS = T25_OUTPUT_FIELDS + ["source_table"]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_table"] = str(path)
    return rows


def build_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    table_paths = [
        Path(args.products_csv),
        Path(args.reddit_csv),
        Path(args.arxiv_csv),
        Path(args.ultra_csv),
    ]
    rows: list[dict[str, Any]] = []
    for path in table_paths:
        rows.extend(_read_rows(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T25 HNR-FDM-lite stage outputs.")
    parser.add_argument("--products-csv", default="experiments/tables/t25_products_recovery_ladder_seed42.csv")
    parser.add_argument("--reddit-csv", default="experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv")
    parser.add_argument("--arxiv-csv", default="experiments/tables/t25_arxiv_sft_v4_teacher_seed42.csv")
    parser.add_argument("--ultra-csv", default="experiments/tables/t25_ultra_dryrun_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t25_hnr_fdm_summary_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t25_hnr_fdm_stage_summary.md")
    args = parser.parse_args()
    rows = build_summary(args)
    output = write_csv(args.csv, rows, SUMMARY_FIELDS)
    promoted = [row for row in rows if str(row.get("promotion_status")) == "promoted"]
    blocked = [row for row in rows if str(row.get("failure_reason", ""))]
    ensure_report(
        args.report,
        [
            "# T25 Shadow-HGC-SFT-HNR-FDM-lite Stage Summary",
            "",
            "## Scope",
            "",
            "- Added scalable HNR/FDM-lite contracts, selectors, guards, and stage runners.",
            "- Existing T24/R-1 paths are not replaced; T25 rows are promoted only when explicit gates pass.",
            "- Exact GCRD TPAMI 2026 numbers are not fabricated; placeholder baseline rows are kept in `baselines/gcrd_tpami26.csv`.",
            "",
            "## Aggregated Rows",
            "",
            *markdown_table(rows, ["dataset", "method", "requested_full_node_ratio", "status", "accuracy", "macro_f1", "promotion_status", "failure_reason", "source_table"]),
            "",
            "## Promoted Rows",
            "",
            *markdown_table(promoted, ["dataset", "method", "requested_full_node_ratio", "accuracy", "macro_f1", "promotion_status"]),
            "",
            "## Rows With Gates Not Met Or Diagnostics",
            "",
            *markdown_table(blocked, ["dataset", "method", "requested_full_node_ratio", "failure_reason"]),
            "",
            "## Required Next Server Commands",
            "",
            "```powershell",
            "& 'C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe' scripts\\run_t25_reddit_hnr_fdm.py --train --epochs 30 --device cuda",
            "& 'C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe' scripts\\run_t25_products_recovery.py --train --epochs 4 --device cuda",
            "& 'C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe' scripts\\run_t25_arxiv_sft_v4.py",
            "& 'C:\\Users\\slian\\anaconda3\\envs\\pytorch\\python.exe' scripts\\run_t25_ultra_dryrun.py --ultra-safe",
            "```",
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
