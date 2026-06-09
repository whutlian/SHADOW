from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv


FIELDS = [
    "dataset",
    "variant",
    "seed",
    "status",
    "kd_gate_passed",
    "kd_skip_reason",
    "teacher_used_for_herding",
    "prototype_mode",
    "teacher_type",
    "accuracy",
    "macro_f1",
    "source_log",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write gated teacher-demand herding/KD rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/tables/teacher_herding_kd_gated_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/teacher_herding_kd_gated_seed42.md")
    args = parser.parse_args()
    rows = []
    for dataset in ["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"]:
        rows.append({
            "dataset": dataset,
            "variant": "teacher_demand_herding",
            "seed": args.seed,
            "status": "skipped_blocked_by_teacher_or_backbone",
            "kd_gate_passed": False,
            "kd_skip_reason": "fullgraph_or_teacher_gate_not_passed_in_clean_sprint",
            "teacher_used_for_herding": False,
            "prototype_mode": "teacher_demand_herding",
            "teacher_type": "none",
            "accuracy": "",
            "macro_f1": "",
            "source_log": "",
        })
        rows.append({
            "dataset": dataset,
            "variant": "kd_v2",
            "seed": args.seed,
            "status": "skipped_blocked_by_teacher_or_backbone",
            "kd_gate_passed": False,
            "kd_skip_reason": "fullgraph_or_teacher_gate_not_passed_in_clean_sprint",
            "teacher_used_for_herding": False,
            "prototype_mode": "",
            "teacher_type": "none",
            "accuracy": "",
            "macro_f1": "",
            "source_log": "",
        })
    output = Path(args.output)
    write_csv(output, rows, FIELDS)
    lines = [
        "# Teacher Herding / KD Gated Seed 42",
        "",
        "No teacher-demand herding or KD v2 row was run as a promoted result because the clean sprint requires a passing fullgraph/teacher gate first. Rows are explicit skipped diagnostics.",
        "",
        "| Dataset | Variant | Status | KD gate | Skip reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['status']} | {row['kd_gate_passed']} | {row['kd_skip_reason']} |")
    lines.extend(["", f"- CSV: `{output}`"])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

