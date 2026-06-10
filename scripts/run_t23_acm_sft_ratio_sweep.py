from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, fvalue, markdown_table, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate T23 ACM SFT condensed ratio sweep.")
    parser.add_argument("--tune-source", default="experiments/tables/t23_acm_sft_tune_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t23_acm_sft_ratio_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t23_acm_sft_ratio_sweep_summary.md")
    args = parser.parse_args()
    tune_rows = read_csv(args.tune_source)
    best = max(tune_rows, key=lambda row: fvalue(row.get("accuracy"))) if tune_rows else {}
    run_sweep = fvalue(best.get("accuracy")) >= 0.93
    rows = []
    if run_sweep:
        for ratio in [0.005, 0.012, 0.024, 0.048, 0.096]:
            rows.append({"dataset": "acm", "ratio": ratio, "ratio_percent": ratio * 100.0, "status": "queued", "reason": "ACM gate passed; run full ratio sweep with train mode"})
    else:
        rows.append(
            {
                "dataset": "acm",
                "ratio": "",
                "ratio_percent": "",
                "status": "skipped_by_gate",
                "best_tune_accuracy": best.get("accuracy", ""),
                "reason": "ACM fullgraph tune did not reach 0.93, so prompt-gated condensed sweep was not run",
            }
        )
    output = write_csv(args.csv, rows)
    ensure_report(
        args.report,
        [
            "# T23 ACM SFT Ratio Sweep",
            "",
            *markdown_table(rows, ["dataset", "ratio_percent", "status", "best_tune_accuracy", "reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
