from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv


def _read_csv(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    with file.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate T0-S condensation recovery after fullgraph parity.")
    parser.add_argument("--fullgraph", default="experiments/tables/t0s_fullgraph_parity_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t0s_condensation_recovery_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t0s_condensation_recovery_summary.md")
    args = parser.parse_args()
    rows = []
    for row in _read_csv(args.fullgraph):
        passed = _truthy(row.get("gate_passed"))
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", 42),
                "fullgraph_variant": row.get("variant", ""),
                "fullgraph_accuracy": row.get("accuracy", ""),
                "fullgraph_gate_passed": passed,
                "condensation_status": "eligible_not_run" if passed else "blocked_by_t0s_fullgraph_gate",
                "promoted": False,
                "ratio_plan": "not_started" if passed else "blocked",
                "reason": "T0-S fullgraph gate passed; condensation sweep must be launched separately" if passed else row.get("blocked_reason") or row.get("reason", ""),
            }
        )
    output = Path(args.output)
    write_csv(output, rows)
    lines = [
        "# T0-S Condensation Recovery Seed 42",
        "",
        *markdown_table(rows, ["dataset", "fullgraph_variant", "fullgraph_accuracy", "fullgraph_gate_passed", "condensation_status", "promoted", "reason"]),
        "",
        "No compressed result is promoted unless the corresponding T0-S fullgraph row passes accuracy and scalability gates.",
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
