from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import markdown_table, write_csv, write_json


FIELDS = [
    "dataset",
    "recovery_row",
    "fullgraph_status",
    "fullgraph_accuracy",
    "fullgraph_macro_f1",
    "selected_blocks",
    "status",
    "promoted",
    "full_to_identity_gap",
    "identity_to_oracle_gap",
    "oracle_to_shadow_gap",
    "full_to_shadow_gap",
    "uses_logits_as_input",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "reason",
]


def _read(path: str | Path) -> list[dict[str, str]]:
    file = Path(path)
    if not file.exists():
        return []
    with file.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_recovery_rows(fullgraph_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fullgraph_rows:
        if row.get("row_kind") not in {"final", "resource_guard"}:
            continue
        eligible = row.get("status") == "promoted"
        if not eligible:
            rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "recovery_row": "recovery_gate",
                    "fullgraph_status": row.get("status", ""),
                    "fullgraph_accuracy": row.get("accuracy", ""),
                    "fullgraph_macro_f1": row.get("macro_f1", ""),
                    "selected_blocks": row.get("selected_blocks", ""),
                    "status": "blocked_by_t2_fullgraph_gate",
                    "promoted": False,
                    "uses_logits_as_input": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "reason": row.get("reason", ""),
                }
            )
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "recovery_row": "identity_condensed_sft_replay",
                "fullgraph_status": row.get("status", ""),
                "fullgraph_accuracy": row.get("accuracy", ""),
                "fullgraph_macro_f1": row.get("macro_f1", ""),
                "selected_blocks": row.get("selected_blocks", ""),
                "status": "completed_diagnostic",
                "promoted": False,
                "full_to_identity_gap": 0.0,
                "identity_to_oracle_gap": "",
                "oracle_to_shadow_gap": "",
                "full_to_shadow_gap": 0.0,
                "uses_logits_as_input": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_e_by_d_materialization": False,
                "reason": "identity replay of the validation-selected T2 SFT table teacher; diagnostic only",
            }
        )
        for recovery in ["prototype_oracle_sft_block_signature", "shadow_condensed_sft_block_signature"]:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "recovery_row": recovery,
                    "fullgraph_status": row.get("status", ""),
                    "fullgraph_accuracy": row.get("accuracy", ""),
                    "fullgraph_macro_f1": row.get("macro_f1", ""),
                    "selected_blocks": row.get("selected_blocks", ""),
                    "status": "eligible_not_run",
                    "promoted": False,
                    "full_to_identity_gap": "",
                    "identity_to_oracle_gap": "",
                    "oracle_to_shadow_gap": "",
                    "full_to_shadow_gap": "",
                    "uses_logits_as_input": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "reason": "fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate T2 condensation recovery diagnostics.")
    parser.add_argument("--fullgraph", default="experiments/tables/t2_sft_fullgraph_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t2_condensation_recovery_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t2_condensation_recovery_summary.md")
    args = parser.parse_args()
    rows = build_recovery_rows(_read(args.fullgraph))
    output = write_csv(args.output, rows, FIELDS)
    write_json(Path(args.output).with_suffix(".json"), {"rows": rows})
    lines = [
        "# T2-SFT-NL Condensation Recovery Gate",
        "",
        "Recovery rows are diagnostic only. Identity replay is materialized for promoted fullgraph rows; prototype/shadow SFT block-signature compression is left as `eligible_not_run` unless launched separately.",
        "",
        *markdown_table(rows, ["dataset", "recovery_row", "fullgraph_accuracy", "status", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
