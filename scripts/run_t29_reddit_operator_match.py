from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.operator_match import build_knn_candidate_edges, fit_operator_match, synthetic_operator_targets
from shadow_hgc.sft.t29_contract import T29_REQUIRED_FIELDS, make_t29_row, ratio_budget


DEFAULT_RATIOS = (0.001, 0.005)
DEFAULT_INITS = ("current_sft_signature_random", "sft_hnr_fdm_hybrid", "ctc_bucket_selection", "sft_bonsai_sketch")
DEFAULT_TOPKS = (4, 8, 16, 32)
DEFAULT_STUDENTS = ("operator_sft_table_head", "weighted_sgc", "weighted_gcn", "weighted_graphsage")


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_omcp_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_reddit_operator_match.py --device cuda --ratios 0.001 0.005 "
        "--prototype-inits current_sft_signature_random sft_hnr_fdm_hybrid ctc_bucket_selection sft_bonsai_sketch "
        "--operator-candidate-builders knn cooccur sketch --operator-topks 4 8 16 32 "
        "--operator-steps 500 1000 2000 --operator-lrs 0.01 0.005 "
        "--students operator_sft_table_head weighted_sgc weighted_gcn weighted_graphsage "
        "--hidden-dims 128 256 512 --epochs 60 120 200 "
        f"--seed {int(seed)} --run-long"
    )


def _method(init: str) -> str:
    if init == "sft_hnr_fdm_hybrid":
        return "reddit_sft_omcp_hnr_hybrid"
    if init == "ctc_bucket_selection":
        return "reddit_sft_omcp_ctc"
    if init == "sft_bonsai_sketch":
        return "reddit_sft_omcp_bonsai"
    return "reddit_sft_omcp_random"


def build_omcp_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    inits = [str(v) for v in _arg(args, "prototype_inits", DEFAULT_INITS)]
    topks = [int(v) for v in _arg(args, "operator_topks", DEFAULT_TOPKS)]
    students = [str(v) for v in _arg(args, "students", DEFAULT_STUDENTS)]
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        budget = ratio_budget("Reddit", ratio)
        for init in inits:
            feature_dim = 8 if smoke else 32
            x0, x1 = synthetic_operator_targets(budget, feature_dim, seed=seed + budget)
            candidates = build_knn_candidate_edges(x0, candidate_topk=min(8 if smoke else 32, max(1, budget - 1)))
            for topk in topks:
                started = time.perf_counter()
                result = fit_operator_match(
                    x0=x0,
                    x1_target=x1,
                    candidate_edge_index=candidates.edge_index,
                    topk=int(topk),
                    steps=8 if smoke else int(_arg(args, "operator_steps", [500])[0]),
                    lr=0.03 if smoke else float(_arg(args, "operator_lrs", [0.01])[0]),
                    seed=seed,
                )
                fit_time = time.perf_counter() - started
                for student in students:
                    rows.append(
                        make_t29_row(
                            dataset="Reddit",
                            method=_method(init),
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            target_prototypes=budget,
                            total_condensed_edges=int(result.edge_index.shape[1]),
                            status="completed_operator_smoke" if smoke else "completed_operator_fit_only",
                            promotion_status="not_promoted",
                            promotion_track="safe_mainline",
                            failure_reason="no_transfer_eval_accuracy",
                            notes="OMCP sparse operator fitted; transfer student accuracy is not reported until long run.",
                            extra={
                                **result.diagnostics,
                                "operator_fit_time": fit_time,
                                "student_model": student,
                                "uses_dense_adjacency": False,
                                "uses_full_edge_index_on_gpu": False,
                                "uses_e_by_d_materialization": False,
                            },
                        )
                    )
                if smoke:
                    break
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_omcp_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_reddit_omcp_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_reddit_omcp_summary.md"),
        [
            "# T29 Reddit OMCP",
            "",
            "- Smoke rows fit sparse operator diagnostics but do not report transfer accuracy.",
            "- Budgets use strict full-node ratio accounting; 0.10% and 0.50% use different node counts.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "actual_condensed_nodes", "operator_edges", "operator_row_sum_error", "operator_negative_weight_count", "student_model", "status", "accuracy", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_omcp_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 Reddit OMCP sparse operator matching.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--prototype-inits", nargs="+", default=list(DEFAULT_INITS))
    parser.add_argument("--operator-candidate-builders", nargs="+", default=["knn"])
    parser.add_argument("--operator-topks", nargs="+", type=int, default=list(DEFAULT_TOPKS))
    parser.add_argument("--operator-steps", nargs="+", type=int, default=[500])
    parser.add_argument("--operator-lrs", nargs="+", type=float, default=[0.01])
    parser.add_argument("--students", nargs="+", default=list(DEFAULT_STUDENTS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[60, 120, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_reddit_omcp_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_reddit_omcp_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
