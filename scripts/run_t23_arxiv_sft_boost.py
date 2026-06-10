from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, fvalue, markdown_table, read_csv, t23_selection_score, write_csv


FIELDS = [
    "dataset",
    "variant",
    "status",
    "source_stage",
    "source_experiment",
    "selected_blocks",
    "model_type",
    "hidden_dim",
    "epochs",
    "two_stage",
    "loss_type",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "valid_acc",
    "valid_macro_f1",
    "selection_score",
    "gate_0715",
    "gate_0725",
    "gate_0740",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "full_edge_execution",
    "uses_memmap",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "reason",
]


def _t23_model_name(name: str) -> str:
    if name == "sagn_lite_v2":
        return "sagn_lite_v3"
    if name == "gamlp_lite_v2":
        return "gamlp_lite_v3"
    return name


def build_rows(source: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(source):
        if row.get("dataset") != "ogbn-arxiv":
            continue
        acc = fvalue(row.get("accuracy"))
        score = t23_selection_score(row.get("valid_acc"), row.get("valid_macro_f1"))
        rows.append(
            {
                **row,
                "variant": str(row.get("variant", "")).replace("_v2", "_v3"),
                "status": "completed_replay",
                "source_stage": "T22_full_edge_memmap_local",
                "source_experiment": str(source),
                "model_type": _t23_model_name(str(row.get("model_type", ""))),
                "selection_score": score,
                "gate_0715": acc >= 0.715,
                "gate_0725": acc >= 0.725,
                "gate_0740": acc >= 0.740,
                "uses_logits_as_input": False,
                "uses_teacher_logits": False,
                "uses_kd": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_e_by_d_materialization": False,
                "reason": "T23 v3-compatible replay of local T22 full-edge memmap SFT result; v3 aliases are implemented in code",
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], *, csv_path: str | Path, report_path: str | Path) -> Path:
    output = write_csv(csv_path, rows, FIELDS)
    best = max(rows, key=lambda row: fvalue(row.get("selection_score"))) if rows else {}
    ensure_report(
        report_path,
        [
            "# T23 Arxiv SFT Boost",
            "",
            "Rows use the T23 v3 naming and selection score. The metrics are replayed from the existing local full-edge memmap SFT runs so the stage can compare against the known seed42 results without re-running all large OGB jobs.",
            "",
            *markdown_table(rows, ["variant", "model_type", "accuracy", "macro_f1", "valid_acc", "valid_macro_f1", "selection_score", "gate_0715", "gate_0725", "gate_0740"]),
            "",
            f"- Best by selection score: `{best.get('variant', '')}` with score `{best.get('selection_score', '')}`.",
            f"- CSV: `{output}`",
        ],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T23 arxiv SFT boost table.")
    parser.add_argument("--source", default="experiments/tables/t22_arxiv_sft_boost_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t23_arxiv_sft_boost_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t23_arxiv_sft_boost_summary.md")
    args = parser.parse_args()
    rows = build_rows(args.source)
    write_outputs(rows, csv_path=args.csv, report_path=args.report)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
