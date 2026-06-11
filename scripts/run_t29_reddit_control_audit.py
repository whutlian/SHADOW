from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t29_contract import T29_REQUIRED_FIELDS, make_t29_row, ratio_budget


DEFAULT_RATIOS = (0.0005, 0.001, 0.0025, 0.005, 0.01)
DEFAULT_METHODS = ("current_sft_signature_random", "sft_hnr_fdm_hybrid", "reddit_random_frozen_init")


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_control_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_reddit_control_audit.py --device cuda "
        "--ratios 0.0005 0.001 0.0025 0.005 0.01 "
        "--methods current_sft_signature_random sft_hnr_fdm_hybrid reddit_random_frozen_init "
        "--hidden-dims 128 256 512 --epochs 30 60 120 --seeds 1 2 3 4 5 42 --run-long"
    )


def _load_refs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        "experiments/tables/t28_reddit_control_audit_seed_sweep.csv",
        "experiments/tables/t27_stc_reddit_0p05_0p1_0p5_percent_seed42.csv",
        "experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv",
        "experiments/tables/t24_reddit_sft_condense_seed42.csv",
    ]:
        for row in read_csv(path):
            item = dict(row)
            item["_source_csv"] = path
            rows.append(item)
    return rows


def _aliases(method: str) -> set[str]:
    if method == "current_sft_signature_random":
        return {"current_sft_signature_random", "SFT-signature random"}
    return {method}


def _match_ref(rows: list[dict[str, Any]], method: str, ratio: float, seed: int) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") not in _aliases(method):
            continue
        if row.get("seed") not in {"", None} and int(float(row.get("seed", 0))) != int(seed):
            continue
        if abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) > 1e-12:
            continue
        if row.get("accuracy") in {"", None}:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: fvalue(row.get("accuracy")))


def build_control_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    methods = [str(v) for v in _arg(args, "methods", DEFAULT_METHODS)]
    seeds = [int(v) for v in _arg(args, "seeds", [_arg(args, "seed", 42)])]
    refs = _load_refs()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for ratio in ratios:
            budget = ratio_budget("Reddit", ratio)
            for method in methods:
                ref = _match_ref(refs, method, ratio, seed)
                if ref:
                    rows.append(
                        make_t29_row(
                            dataset="Reddit",
                            method=method,
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            target_prototypes=budget,
                            accuracy=ref.get("accuracy", ""),
                            macro_f1=ref.get("macro_f1", ""),
                            valid_acc=ref.get("valid_acc", ""),
                            predicted_classes=ref.get("predicted_classes", ref.get("predicted_class_count", "")),
                            status="completed_reference",
                            promotion_status="not_promoted",
                            promotion_track="safe_mainline",
                            failure_reason="control_reference_not_new_t29_method",
                            notes="T29 control audit imports prior metric but fixes full-node ratio budget accounting.",
                            source_table=ref.get("_source_csv", ""),
                        )
                    )
                else:
                    rows.append(
                        make_t29_row(
                            dataset="Reddit",
                            method=method,
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            target_prototypes=budget,
                            status="completed_smoke" if bool(_arg(args, "smoke", False)) else "server_ready_not_run",
                            promotion_status="not_promoted",
                            promotion_track="safe_mainline",
                            failure_reason="smoke_only_no_training" if bool(_arg(args, "smoke", False)) else "server_command_required",
                            notes="Control row with ratio-correct budget; no new T29 training result.",
                        )
                    )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_control_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_reddit_control_audit_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_reddit_control_audit_summary.md"),
        [
            "# T29 Reddit Control Audit",
            "",
            "- Control rows preserve prior metrics but fix full-node ratio budgets.",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "actual_condensed_nodes", "accuracy", "macro_f1", "status", "source_table"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_control_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 Reddit control audit with ratio-correct budgets.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--epochs", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_reddit_control_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_reddit_control_audit_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
