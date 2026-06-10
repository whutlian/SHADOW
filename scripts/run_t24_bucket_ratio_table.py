from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.ratio.scale_bucket import fixed_bucket_main_rows, ratio_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T24 bucket ratio policy table.")
    parser.add_argument("--ratio-mode", default="full_node", choices=["full_node"])
    parser.add_argument("--scale-bucket", default="auto", choices=["auto", "medium", "large", "ultra"])
    parser.add_argument("--ratio-preset", default="bucket_default", choices=["bucket_default", "bucket_sweep"])
    parser.add_argument("--csv", default="experiments/tables/t24_bucket_ratio_policy_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_bucket_ratio_policy_summary.md")
    args = parser.parse_args()
    rows = fixed_bucket_main_rows()
    if args.scale_bucket != "auto":
        rows = [
            {
                "dataset": f"{args.scale_bucket}_bucket",
                "nodes": "",
                "scale_bucket": args.scale_bucket,
                "ratio_mode": args.ratio_mode,
                "main_ratio": ratio_preset(scale_bucket=args.scale_bucket, preset="bucket_default")[0],
                "sweep_ratios": ",".join(str(value) for value in ratio_preset(scale_bucket=args.scale_bucket, preset="bucket_sweep")),
            }
        ]
    output = write_csv(args.csv, rows)
    ensure_report(
        args.report,
        [
            "# T24 Bucket Ratio Policy",
            "",
            *markdown_table(rows, ["dataset", "scale_bucket", "ratio_mode", "main_ratio", "sweep_ratios"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
