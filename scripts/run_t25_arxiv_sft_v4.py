from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t24_arxiv_sft_v4 import build_rows as build_t24_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t25_contract import T25_OUTPUT_FIELDS, make_t25_row


FIELDS = T25_OUTPUT_FIELDS + ["variant", "selected_blocks", "teacher_gate_A1", "teacher_gate_A2", "teacher_gate_A3", "promotion_reason"]


def build_rows(seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in build_t24_rows():
        acc = source.get("accuracy", "")
        macro = source.get("macro_f1", "")
        predicted = source.get("predicted_class_count", "")
        acc_value = float(acc) if acc != "" else 0.0
        row = make_t25_row(
            dataset="ogbn-arxiv",
            method="arxiv_sft_v4_teacher",
            requested_full_node_ratio=0.005,
            original_total_nodes=169_343,
            target_prototypes=0,
            shadow_nodes=0,
            total_condensed_edges=0,
            seed=seed,
            accuracy=acc,
            macro_f1=macro,
            predicted_classes=predicted,
            status=source.get("status", ""),
            promotion_status="promoted" if acc_value >= 0.715 else "not_promoted",
            promotion_reason="passed_A1_teacher_gate" if acc_value >= 0.715 else "A1_teacher_gate_not_met",
            notes="teacher-first gate; condensation is blocked until A1 passes",
            full_edge_scans=source.get("full_edge_scans", ""),
            cache_bytes=source.get("cache_bytes", ""),
            uses_dense_p2=source.get("uses_dense_p2", False),
            uses_e_by_d_materialization=source.get("uses_e_by_d", False),
        )
        row["variant"] = source.get("variant", "")
        row["selected_blocks"] = source.get("selected_blocks", "")
        row["teacher_gate_A1"] = acc_value >= 0.715
        row["teacher_gate_A2"] = acc_value >= 0.725
        row["teacher_gate_A3"] = acc_value >= 0.740
        if acc_value < 0.715:
            row["failure_reason"] = "A1_teacher_gate_not_met"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T25 arxiv SFT-v4 teacher gate table.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="experiments/tables/t25_arxiv_sft_v4_teacher_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t25_arxiv_sft_v4_teacher.md")
    args = parser.parse_args()
    rows = build_rows(seed=int(args.seed))
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T25 Arxiv SFT-v4 Teacher Gate",
            "",
            "- Condensation remains blocked until A1 accuracy >= 0.715.",
            "",
            *markdown_table(rows, ["variant", "status", "accuracy", "macro_f1", "predicted_classes", "teacher_gate_A1", "teacher_gate_A2", "teacher_gate_A3", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
