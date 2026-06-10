from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, fvalue, markdown_table, read_csv, write_csv


def build_rows(source: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(source):
        if row.get("dataset") != "acm":
            continue
        acc = fvalue(row.get("accuracy"))
        rows.append(
            {
                **row,
                "status": "completed_replay",
                "source_stage": "T22_local_acm_tune",
                "source_experiment": str(source),
                "gate_093": acc >= 0.93,
                "run_condensed_sweep": acc >= 0.93,
                "reason": "T23 ACM tune replay; condensed sweep is gated on matching/improving current best",
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], *, csv_path: str | Path, report_path: str | Path) -> Path:
    output = write_csv(csv_path, rows)
    best = max(rows, key=lambda row: fvalue(row.get("accuracy"))) if rows else {}
    ensure_report(
        report_path,
        [
            "# T23 ACM SFT Tune",
            "",
            *markdown_table(rows, ["variant", "accuracy", "macro_f1", "valid_acc", "gate_093", "run_condensed_sweep"]),
            "",
            f"- Best ACM tune row: `{best.get('variant', '')}` accuracy `{best.get('accuracy', '')}`.",
            f"- Condensed ACM sweep gated on >=0.93: `{bool(best.get('run_condensed_sweep', False))}`.",
            f"- CSV: `{output}`",
        ],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T23 ACM tune table.")
    parser.add_argument("--source", default="experiments/tables/t22_acm_sft_tune_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t23_acm_sft_tune_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t23_acm_sft_tune_summary.md")
    args = parser.parse_args()
    rows = build_rows(args.source)
    write_outputs(rows, csv_path=args.csv, report_path=args.report)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
