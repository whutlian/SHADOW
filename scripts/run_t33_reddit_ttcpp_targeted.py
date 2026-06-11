from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t33_reddit_ratio_curve import build_ratio_curve_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS


def write_outputs(args: argparse.Namespace) -> Path:
    args.ratios = [float(args.ratio)]
    rows = build_ratio_curve_rows(args)
    csv_path = write_csv(args.csv, rows, T33_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T33 Reddit TTC++ Targeted 0.50%",
            "",
            *markdown_table(rows, ["method", "accuracy", "macro_f1", "valid_acc", "hidden_dim", "epochs", "soft_temperature", "lambda_hard", "lambda_prior", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 Reddit targeted 0.50 TTC++ sweep.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.005)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"]))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--teacher-ensemble-cache-dir", default="experiments/cache/t32_reddit_teacher_ensemble_seed42")
    parser.add_argument("--teacher-model-type", default="sagn_lite_v4")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-dropout", type=float, default=0.3)
    parser.add_argument("--teacher-num-layers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--methods", nargs="+", default=["reddit_ttcpp_sagn_table_student"])
    parser.add_argument("--teacher-cache-mode", default="dense_fp16")
    parser.add_argument("--budget-policy", default="ratio_adaptive_v2")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[4.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.25])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.02])
    parser.add_argument("--lambda-conf", type=float, default=0.0)
    parser.add_argument("--lambda-mix", type=float, default=0.0)
    parser.add_argument("--student-model-type", default="sagn_lite_v4")
    parser.add_argument("--student-lr", type=float, default=0.003)
    parser.add_argument("--student-batch-size", type=int, default=2048)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256])
    parser.add_argument("--epochs", nargs="+", type=int, default=[260])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--checkpoint-selection", default="best_valid_acc")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_reddit_ttcpp_targeted_0p50.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_reddit_ttcpp_targeted_0p50.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
