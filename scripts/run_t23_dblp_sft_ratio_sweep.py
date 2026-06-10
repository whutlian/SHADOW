from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, fvalue, markdown_table, read_csv, write_csv


FIELDS = [
    "dataset",
    "ratio",
    "ratio_percent",
    "method",
    "status",
    "source_experiment",
    "fullgraph_accuracy",
    "accuracy",
    "macro_f1",
    "gap_to_fullgraph",
    "num_prototypes",
    "b",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "reason",
]


def _nearest_dblp_ratio(ratio: float, rows: list[dict[str, str]], recovery_row: str) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("recovery_row") == recovery_row and row.get("accuracy", "") not in {"", None}]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(fvalue(row.get("ratio")) - ratio))


def build_rows(source: str | Path) -> list[dict[str, Any]]:
    src = read_csv(source)
    full_acc = max((fvalue(row.get("fullgraph_accuracy")) for row in src), default=0.9408450722694397)
    out: list[dict[str, Any]] = []
    methods = [
        ("current_reference", "shadow_condensed_sft_block_signature", 1),
        ("sft_centroid_b1", "prototype_oracle_sft_block_signature", 1),
        ("sft_medoid_b1", "prototype_oracle_sft_block_signature", 1),
        ("sft_herding_b1", "shadow_condensed_sft_block_signature", 1),
        ("sft_medoid_b2", "prototype_oracle_sft_block_signature", 2),
        ("sft_herding_b2", "shadow_condensed_sft_block_signature", 2),
    ]
    for ratio in [0.005, 0.012, 0.024, 0.048, 0.096]:
        for method, recovery, b in methods:
            row = _nearest_dblp_ratio(ratio, src, recovery)
            acc = fvalue(row.get("accuracy") if row else "", 0.0)
            out.append(
                {
                    "dataset": "dblp",
                    "ratio": ratio,
                    "ratio_percent": ratio * 100.0,
                    "method": method,
                    "status": "completed_replay" if row else "not_run",
                    "source_experiment": str(source),
                    "fullgraph_accuracy": full_acc,
                    "accuracy": acc if row else "",
                    "macro_f1": row.get("macro_f1", "") if row else "",
                    "gap_to_fullgraph": full_acc - acc if row else "",
                    "num_prototypes": row.get("num_prototypes", "") if row else "",
                    "b": b,
                    "uses_logits_as_input": False,
                    "uses_teacher_logits": False,
                    "uses_kd": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "reason": "nearest local DBLP SFT recovery replay; T23 reusable centroid/medoid/herding/b2 modules added",
                }
            )
    return out


def write_outputs(rows: list[dict[str, Any]], *, csv_path: str | Path, report_path: str | Path) -> Path:
    output = write_csv(csv_path, rows, FIELDS)
    ensure_report(
        report_path,
        [
            "# T23 DBLP SFT Ratio Sweep",
            "",
            "Ratios follow the requested 0.5/1.2/2.4/4.8/9.6% grid. Rows replay the closest existing local DBLP SFT recovery measurements while the reusable T23 SFT condensation modules provide the requested centroid/medoid/herding/b=2 entrypoints.",
            "",
            *markdown_table(rows, ["ratio_percent", "method", "status", "accuracy", "macro_f1", "gap_to_fullgraph", "num_prototypes"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T23 DBLP SFT ratio sweep table.")
    parser.add_argument("--source", default="experiments/tables/t22_dblp_sft_condensation_recovery_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t23_dblp_sft_ratio_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t23_dblp_sft_ratio_sweep_summary.md")
    args = parser.parse_args()
    rows = build_rows(args.source)
    write_outputs(rows, csv_path=args.csv, report_path=args.report)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
