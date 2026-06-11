from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from scripts.run_t32_reddit_ttcpp import build_reddit_ttcpp_rows
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS, apply_t33_promotion_guard, make_t33_row, reddit_gate_status
from shadow_hgc.sft.ttcpp_ratio_curve import aggregate_ratio_curve


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def t32_to_t33_row(row: dict[str, Any], *, teacher_cache_mode: str, checkpoint_selection: str = "best_valid_acc") -> dict[str, Any]:
    ratio = _f(row.get("requested_full_node_ratio"))
    accuracy = row.get("accuracy", "")
    macro_f1 = row.get("macro_f1", "")
    status, reason = ("not_promoted", "")
    if accuracy not in {"", None} and macro_f1 not in {"", None}:
        status, reason = reddit_gate_status(ratio=ratio, accuracy=float(accuracy), macro_f1=float(macro_f1))
    if str(row.get("failure_reason", "")) and status != "promoted":
        reason = str(row.get("failure_reason", reason))
    cache_bytes = row.get("teacher_cache_bytes", row.get("cache_bytes", ""))
    method = str(row.get("method", ""))
    if method == "reddit_ttcpp_ratio_adaptive_core40":
        method = "reddit_ttcpp_ratio_adaptive_v2"
    t33 = make_t33_row(
        dataset="Reddit",
        method=method,
        seed=int(_f(row.get("seed", 42), 42)),
        requested_full_node_ratio=ratio,
        total_condensed_nodes=int(_f(row.get("condensed_nodes", row.get("total_condensed_nodes", 0)))),
        shadow_nodes=0,
        condensed_edges=0,
        accuracy=accuracy,
        macro_f1=macro_f1,
        valid_acc=row.get("valid_acc", ""),
        predicted_classes=row.get("predicted_classes", ""),
        status=row.get("status", "completed_long"),
        failure_reason=reason,
        promotion_track="sota_chase",
        promotion_status=status,
        teacher_cache_mode=teacher_cache_mode,
        teacher_cache_bytes=cache_bytes,
        teacher_ensemble_size=row.get("teacher_ensemble_size", 1),
        teacher_accuracy=row.get("teacher_accuracy", ""),
        teacher_valid_acc=row.get("teacher_valid_acc", ""),
        teacher_temperature=row.get("teacher_temperature", ""),
        teacher_entropy_mean=row.get("teacher_entropy_mean", ""),
        teacher_disagreement_mean=row.get("teacher_disagreement_mean", ""),
        teacher_pairwise_kl_mean=row.get("teacher_pairwise_kl_mean", ""),
        teacher_cache_duplicate_detected=row.get("teacher_cache_duplicate_detected", False),
        uses_teacher_logits=True,
        uses_teacher_probs=True,
        uses_logits_as_input=False,
        uses_kd=True,
        candidate_nodes_mode=row.get("candidate_nodes", "all") or "all",
        budget_policy=row.get("budget_policy", "ratio_adaptive_v2") or "ratio_adaptive_v2",
        selected_soft_prior_kl_to_teacher_prior=row.get("selected_soft_prior_kl", ""),
        entropy_bucket_coverage=row.get("entropy_bucket_coverage", ""),
        margin_bucket_coverage=row.get("margin_bucket_coverage", ""),
        signature_bucket_coverage=row.get("degree_bucket_coverage", ""),
        selected_rows_per_class_min=row.get("class_coverage_min", ""),
        selected_rows_per_class_median=row.get("class_coverage_median", ""),
        selected_rows_per_class_max=row.get("class_coverage_max", ""),
        student_model=row.get("student_model", ""),
        hidden_dim=row.get("hidden_dim", ""),
        epochs=row.get("epochs", ""),
        soft_temperature=row.get("soft_temperature", ""),
        lambda_soft=row.get("lambda_soft", 1.0),
        lambda_hard=row.get("lambda_hard", ""),
        lambda_prior=row.get("lambda_prior", ""),
        lambda_mix=row.get("lambda_mix", ""),
        checkpoint_selection=checkpoint_selection,
        precompute_time=row.get("precompute_time", ""),
        selection_time=row.get("selection_time", ""),
        training_time=row.get("training_time", ""),
        peak_cpu_ram=row.get("peak_cpu_ram", ""),
        peak_gpu_ram=row.get("peak_gpu_ram", ""),
        cache_bytes=row.get("cache_bytes", ""),
        notes=row.get("notes", ""),
    )
    return apply_t33_promotion_guard(t33)


def build_ratio_curve_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in [int(v) for v in _arg(args, "seeds", [int(_arg(args, "seed", 42))])]:
        local = argparse.Namespace(**vars(args))
        local.seed = seed
        t32_rows = build_reddit_ttcpp_rows(local)
        out.extend(
            [
                t32_to_t33_row(
                    row,
                    teacher_cache_mode=str(_arg(args, "teacher_cache_mode", "dense_fp16")),
                    checkpoint_selection=str(_arg(args, "checkpoint_selection", "best_valid_acc")),
                )
                for row in t32_rows
            ]
        )
    return out


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_ratio_curve_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t33_reddit_ttcpp_ratio_curve.csv"), rows, T33_REQUIRED_FIELDS)
    agg = aggregate_ratio_curve(rows)
    write_csv(_arg(args, "multiseed_csv", "experiments/tables/t33_reddit_ttcpp_multiseed.csv"), agg)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t33_reddit_ratio_curve_summary.md"),
        [
            "# T33 Reddit TTC++ Ratio Curve",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "teacher_cache_mode", "promotion_status", "failure_reason"]),
            "",
            "## Aggregate",
            "",
            *markdown_table(agg, ["method", "requested_full_node_ratio", "seed_count", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "valid_test_gap_mean"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 Reddit TTC++ ratio curve.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
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
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--methods", nargs="+", default=["reddit_ttcpp_gamlp_table_student", "reddit_ttcpp_sagn_table_student"])
    parser.add_argument("--teacher-cache-mode", default="dense_fp16")
    parser.add_argument("--candidate-nodes", default="all")
    parser.add_argument("--budget-policy", default="ratio_adaptive_v2")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.25])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.02])
    parser.add_argument("--lambda-conf", type=float, default=0.0)
    parser.add_argument("--lambda-mix", type=float, default=0.0)
    parser.add_argument("--student-model-type", default="sagn_lite_v4")
    parser.add_argument("--student-lr", type=float, default=0.003)
    parser.add_argument("--student-batch-size", type=int, default=2048)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256])
    parser.add_argument("--epochs", nargs="+", type=int, default=[220])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--checkpoint-selection", default="best_valid_acc")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_reddit_ttcpp_ratio_curve.csv")
    parser.add_argument("--multiseed-csv", default="experiments/tables/t33_reddit_ttcpp_multiseed.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_reddit_ratio_curve_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
