from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t28_contract import ARXIV_TEACHER_FIELDS, apply_t28_promotion_guard, make_arxiv_teacher_row


DEFAULT_MODELS: tuple[str, ...] = ("gcn_res", "graphsage", "gat_lite")


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_gnn_upper_bound_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    models = list(_arg(args, "models", DEFAULT_MODELS))
    smoke = bool(_arg(args, "smoke", False))
    run_long = bool(_arg(args, "run_long", False))
    status = "completed_smoke" if smoke else "server_ready_not_run"
    if run_long:
        status = "server_ready_not_run"
    rows: list[dict[str, Any]] = []
    for model in models:
        method = f"arxiv_{model}_cns" if bool(_arg(args, "enable_cns", False)) else f"arxiv_{model}"
        row = make_arxiv_teacher_row(
            method=method,
            seed=seed,
            status=status,
            promotion_status="not_promoted",
            failure_reason="upper_bound_diagnostic_not_promoted",
            notes=(
                "Fullgraph GNN teacher upper-bound diagnostic. It is not a scalable main row "
                "and is never promoted by the T28 guard."
            ),
            uses_cns_postprocess=bool(_arg(args, "enable_cns", False)),
            uses_fullgraph_gnn_teacher=True,
            uses_gnn_hidden_blocks=bool(_arg(args, "export_hidden_blocks", False)),
            upper_bound_diagnostic=True,
            cns_correction_alpha=0.4 if bool(_arg(args, "enable_cns", False)) else "",
            cns_smoothing_alpha=0.4 if bool(_arg(args, "enable_cns", False)) else "",
            cns_correction_steps=20 if bool(_arg(args, "enable_cns", False)) else "",
            cns_smoothing_steps=20 if bool(_arg(args, "enable_cns", False)) else "",
            cns_autoscale=True if bool(_arg(args, "enable_cns", False)) else "",
        )
        rows.append(apply_t28_promotion_guard(row, dataset_gate_passed=False))
    return rows


def write_gnn_outputs(args: argparse.Namespace) -> Path:
    rows = build_gnn_upper_bound_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t28_arxiv_gnn_teacher_upper_bound_seed42.csv"), rows, ARXIV_TEACHER_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t28_arxiv_gnn_teacher_upper_bound_summary.md"),
        [
            "# T28 Arxiv GNN Teacher Upper-Bound",
            "",
            "- Rows are diagnostic fullgraph GNN teachers and are not scalable-main promotions.",
            "- Hidden blocks, if exported, are marked explicitly.",
            "",
            *markdown_table(rows, ["method", "status", "uses_fullgraph_gnn_teacher", "uses_gnn_hidden_blocks", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T28 arxiv fullgraph GNN teacher upper-bound declarations.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--export-hidden-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t28_arxiv_gnn_teacher_upper_bound_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_arxiv_gnn_teacher_upper_bound_summary.md")
    args = parser.parse_args()
    csv_path = write_gnn_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
