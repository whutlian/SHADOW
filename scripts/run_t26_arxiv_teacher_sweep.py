from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t25_arxiv_sft_v4 import build_rows as build_t25_arxiv_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row


FIELDS = T26_REQUIRED_FIELDS + ["variant", "selected_blocks", "teacher_gate_A1", "teacher_gate_A2", "teacher_gate_A3", "condensation_status"]


def build_rows(seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in build_t25_arxiv_rows(seed=int(seed)):
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
            notes="Condensation rows remain blocked until the arxiv teacher reaches A1 >= 0.715.",
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
    parser.add_argument("--csv", default="experiments/tables/t26_arxiv_teacher_sweep_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_arxiv_teacher_notes.md")
    args = parser.parse_args()
    rows = build_rows(seed=int(args.seed))
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
