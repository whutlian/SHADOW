from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t25_arxiv_sft_v4 import build_rows as build_t25_arxiv_rows
from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row


FIELDS = T26_REQUIRED_FIELDS + ["variant", "selected_blocks", "teacher_gate_A1", "teacher_gate_A2", "teacher_gate_A3", "condensation_status"]


def _source_rows(seed: int, actual_source_csv: str | Path | None) -> list[dict[str, Any]]:
    if actual_source_csv not in {"", None}:
        source_path = Path(actual_source_csv)
        if source_path.exists():
            rows = [row for row in read_csv(source_path) if row.get("dataset") == "ogbn-arxiv"]
            if rows:
                for row in rows:
                    row.setdefault("source_table", str(source_path))
                return rows
    return build_t25_arxiv_rows(seed=int(seed))


def build_rows(seed: int = 42, actual_source_csv: str | Path | None = "experiments/tables/t26_arxiv_teacher_actual_seed42.csv") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _source_rows(int(seed), actual_source_csv):
        acc = source.get("accuracy", "")
        macro = source.get("macro_f1", "")
        predicted = source.get("predicted_class_count", source.get("predicted_classes", ""))
        acc_value = 0.0 if acc in {"", None} else float(acc)
        gate_a1 = bool(acc_value >= 0.715)
        condensation_status = "ready_for_condensation" if gate_a1 else "blocked_by_teacher_gate"
        row = make_t26_row(
            dataset="ogbn-arxiv",
            method="arxiv_teacher_sweep",
            requested_full_node_ratio=0.0,
            original_total_nodes=169_343,
            target_prototypes=0,
            shadow_nodes=0,
            total_condensed_edges=0,
            seed=int(seed),
            accuracy=acc,
            macro_f1=macro,
            predicted_classes=predicted,
            status=source.get("status", "ready_not_rerun"),
            promotion_status="not_promoted",
            promotion_reason="teacher_first_gate_only" if gate_a1 else "A1_teacher_gate_not_met",
            failure_reason="" if gate_a1 else "A1_teacher_gate_not_met",
            notes="Arxiv teacher-first gate row; actual source is used when experiments/tables/t26_arxiv_teacher_actual_seed42.csv exists.",
            full_edge_scans=source.get("full_edge_scans", ""),
            cache_bytes=source.get("cache_bytes", ""),
            uses_dense_p2=source.get("uses_dense_p2", False),
            uses_e_by_d_materialization=source.get("uses_e_by_d_materialization", source.get("uses_e_by_d", False)),
            variant=source.get("variant", ""),
            selected_blocks=source.get("selected_blocks", ""),
            teacher_gate_A1=gate_a1,
            teacher_gate_A2=bool(acc_value >= 0.725),
            teacher_gate_A3=bool(acc_value >= 0.740),
            condensation_status=condensation_status,
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T26 arxiv teacher-first sweep.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actual-source-csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t26_arxiv_teacher_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_arxiv_teacher_notes.md")
    args = parser.parse_args()
    rows = build_rows(seed=int(args.seed), actual_source_csv=args.actual_source_csv)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Arxiv Teacher Notes",
            "",
            "- Teacher-first gate A1 is accuracy >= 0.715.",
            "- Condensation rows are blocked while A1 is not met.",
            "",
            *markdown_table(rows, ["variant", "status", "accuracy", "macro_f1", "predicted_class_count", "teacher_gate_A1", "condensation_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
