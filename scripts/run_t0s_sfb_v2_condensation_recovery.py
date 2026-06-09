from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv


def _read(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    with file.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate SFB-v2 condensation recovery.")
    parser.add_argument("--fullgraph", default="experiments/tables/t0s_sfb_v2_fullgraph_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t0s_sfb_v2_condensation_recovery_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t0s_sfb_v2_condensation_recovery_summary.md")
    args = parser.parse_args()
    rows = []
    best_by_dataset: dict[str, dict] = {}
    for row in _read(args.fullgraph):
        if row.get("status") not in {"completed", "diagnostic_existing"}:
            continue
        acc = float(row["accuracy"]) if row.get("accuracy") else -1.0
        prev = best_by_dataset.get(row["dataset"])
        if prev is None or acc > float(prev.get("accuracy") or -1):
            best_by_dataset[row["dataset"]] = row
    for dataset, row in best_by_dataset.items():
        passed = _truthy(row.get("gate_acc_passed"))
        for recovery in ["identity_condensed", "prototype_oracle", "shadow_hgc_sfb_v2_signal"]:
            rows.append(
                {
                    "dataset": dataset,
                    "fullgraph_variant": row.get("variant", ""),
                    "fullgraph_acc": row.get("accuracy", ""),
                    "recovery_row": recovery,
                    "status": "eligible_not_run" if passed else "blocked_by_sfb_v2_fullgraph_gate",
                    "promoted": False,
                    "full_to_identity_gap": "",
                    "identity_to_oracle_gap": "",
                    "oracle_to_shadow_gap": "",
                    "full_to_shadow_gap": "",
                    "reason": "fullgraph gate passed; launch condensation separately" if passed else "fullgraph gate did not pass",
                }
            )
    output = Path(args.output)
    write_csv(output, rows)
    lines = ["# T0-S SFB-v2 Condensation Recovery Seed 42", "", *markdown_table(rows, ["dataset", "fullgraph_variant", "fullgraph_acc", "recovery_row", "status", "promoted", "reason"]), "", f"- CSV: `{output}`"]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
