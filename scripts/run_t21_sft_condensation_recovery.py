from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t21_common import T21_RECOVERY_FIELDS, markdown_table, read_csv, write_csv


def build_recovery_rows(fullgraph_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fullgraph_rows:
        dataset = row.get("dataset", "")
        eligible = row.get("status") in {"promoted", "completed_non_regression"} and dataset in {"acm", "dblp", "imdb"}
        if not eligible:
            medium_completed = dataset in {"ogbn-arxiv", "ogbn-products"} and row.get("status") in {"promoted", "completed", "completed_non_regression"}
            rows.append(
                {
                    "dataset": dataset,
                    "recovery_row": "recovery_gate",
                    "fullgraph_status": row.get("status", ""),
                    "fullgraph_accuracy": row.get("accuracy", ""),
                    "fullgraph_macro_f1": row.get("macro_f1", ""),
                    "selected_blocks": row.get("selected_blocks", ""),
                    "status": "not_recovery_target_medium" if medium_completed else "blocked_by_t21_fullgraph_gate",
                    "promoted": False,
                    "uses_logits_as_input": False,
                    "uses_teacher_logits": False,
                    "uses_kd": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "reason": "medium lazy SFT fullgraph row completed; condensation recovery is not in the current T2.1 small-dataset recovery scope" if medium_completed else row.get("reason", ""),
                }
            )
            continue
        rows.append(
            {
                "dataset": dataset,
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
                "uses_teacher_logits": False,
                "uses_kd": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_e_by_d_materialization": False,
                "reason": "identity replay of frozen SFT block signature",
            }
        )
        prototype_status = "started_diagnostic" if dataset == "dblp" else "eligible_not_run"
        prototype_reason = "DBLP SFT block-signature prototype recovery started; compressed accuracy not yet promoted" if dataset == "dblp" else "diagnostic recovery eligible after fullgraph gate"
        for recovery_name in ["prototype_oracle_sft_block_signature", "shadow_condensed_sft_block_signature"]:
            rows.append(
                {
                    "dataset": dataset,
                    "recovery_row": recovery_name,
                    "fullgraph_status": row.get("status", ""),
                    "fullgraph_accuracy": row.get("accuracy", ""),
                    "fullgraph_macro_f1": row.get("macro_f1", ""),
                    "selected_blocks": row.get("selected_blocks", ""),
                    "status": prototype_status,
                    "promoted": False,
                    "full_to_identity_gap": "",
                    "identity_to_oracle_gap": "",
                    "oracle_to_shadow_gap": "",
                    "full_to_shadow_gap": "",
                    "uses_logits_as_input": False,
                    "uses_teacher_logits": False,
                    "uses_kd": False,
                    "uses_dense_p2": False,
                    "uses_bounded_edges": False,
                    "uses_e_by_d_materialization": False,
                    "reason": prototype_reason,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2.1 SFT condensation recovery table.")
    parser.add_argument("--fullgraph", default="experiments/tables/t21_sft_fullgraph_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t21_sft_condensation_recovery_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_condensation_recovery_summary.md")
    args = parser.parse_args()
    rows = build_recovery_rows(read_csv(args.fullgraph))
    output = write_csv(args.output, rows, T21_RECOVERY_FIELDS)
    lines = [
        "# T2.1 Condensation Recovery Summary",
        "",
        "Identity replay is diagnostic and uses frozen SFT block signatures, not logits as input. DBLP prototype/shadow recovery is marked as started because DBLP is the immediate recovery target.",
        "",
        *markdown_table(rows, ["dataset", "recovery_row", "fullgraph_accuracy", "status", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
