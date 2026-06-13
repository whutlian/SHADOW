from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t33_reddit_ratio_curve import build_ratio_curve_rows as build_t33_ratio_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.stt_cache import estimate_stt_cache_bytes
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, apply_t34_promotion_guard, make_t34_row, reddit_stt_gate_status


T34_TO_T33_METHOD = {
    "reddit_stt_gamlp_ratio_v2": "reddit_ttcpp_gamlp_table_student",
    "reddit_stt_sagn_ratio_v2": "reddit_ttcpp_sagn_table_student",
    "reddit_stt_ensemble_ratio_v2": "reddit_ttcpp_teacher_ensemble_coverage_boundary",
    "reddit_stt_gamlp_sagn_ensemble_student": "reddit_ttcpp_teacher_ensemble_coverage_boundary",
}
T33_TO_T34_METHOD = {
    "reddit_ttcpp_gamlp_table_student": "reddit_stt_gamlp_ratio_v2",
    "reddit_ttcpp_sagn_table_student": "reddit_stt_sagn_ratio_v2",
    "reddit_ttcpp_teacher_ensemble_coverage_boundary": "reddit_stt_ensemble_ratio_v2",
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _method_to_t33(method: str) -> str:
    return T34_TO_T33_METHOD.get(str(method), str(method).replace("reddit_stt", "reddit_ttcpp"))


def _method_to_t34(method: str) -> str:
    if method in T33_TO_T34_METHOD:
        return T33_TO_T34_METHOD[method]
    if method.endswith("_dense_fp16") or method.endswith("_topk4_fp16") or method.endswith("_topk8_fp16"):
        base, mode = method.rsplit("_", 1)
        return f"{_method_to_t34(base)}_{mode}"
    return str(method).replace("reddit_ttcpp", "reddit_stt")


def t33_to_t34_reddit_row(row: dict[str, Any], *, method: str | None = None, teacher_cache_mode: str = "dense_fp16") -> dict[str, Any]:
    ratio = _f(row.get("requested_full_node_ratio"))
    acc = row.get("accuracy", "")
    macro = row.get("macro_f1", "")
    promotion_status = "not_promoted"
    reason = str(row.get("failure_reason", ""))
    if acc not in {"", None} and macro not in {"", None} and str(row.get("status", "")) != "blocked":
        promotion_status, gate_reason = reddit_stt_gate_status(ratio=ratio, accuracy=float(acc), macro_f1=float(macro))
        reason = gate_reason
    if str(row.get("status", "")) == "blocked" and not reason:
        reason = "blocked_source_row"
    estimates = estimate_stt_cache_bytes(num_nodes=232_965, num_classes=41, mode=teacher_cache_mode)
    cache_bytes = row.get("teacher_cache_bytes", "") or estimates["teacher_cache_bytes"]
    valid_acc = row.get("valid_acc", "")
    valid_test_gap = abs(_f(valid_acc) - _f(acc)) if valid_acc not in {"", None} and acc not in {"", None} else ""
    out = make_t34_row(
        dataset="Reddit",
        method=method or _method_to_t34(str(row.get("method", ""))),
        seed=_i(row.get("seed", 42), 42),
        requested_full_node_ratio=ratio,
        condensed_nodes=_i(row.get("total_condensed_nodes", row.get("condensed_nodes", 0))),
        condensed_edges=_i(row.get("condensed_edges", 0)),
        accuracy=acc,
        macro_f1=macro,
        valid_acc=valid_acc,
        status=row.get("status", "completed_long"),
        promotion_track="sota_chase",
        promotion_status=promotion_status,
        failure_reason=reason,
        teacher_method=row.get("teacher_method", row.get("teacher_model_type", "")),
        teacher_accuracy=row.get("teacher_accuracy", ""),
        teacher_valid_acc=row.get("teacher_valid_acc", ""),
        teacher_cache_mode=teacher_cache_mode,
        teacher_cache_bytes=cache_bytes,
        teacher_dense_cache_bytes_diagnostic=estimates["teacher_dense_cache_bytes_diagnostic"],
        cache_compression_ratio=estimates["cache_compression_ratio"],
        uses_teacher_probs=True,
        uses_teacher_logits=True,
        uses_logits_as_input=False,
        uses_teacher_probs_as_input=False,
        soft_target_only=True,
        uses_valid_labels_as_input=False,
        uses_test_labels_as_input=False,
        student_model=row.get("student_model", ""),
        hidden_dim=row.get("hidden_dim", ""),
        epochs=row.get("epochs", ""),
        soft_temperature=row.get("soft_temperature", row.get("temperature", "")),
        lambda_soft=row.get("lambda_soft", 1.0),
        lambda_hard=row.get("lambda_hard", ""),
        lambda_prior=row.get("lambda_prior", ""),
        lambda_cover=0.0,
        lambda_calib=0.0,
        lambda_mix=row.get("lambda_mix", 0.0),
        selection_time=row.get("selection_time", ""),
        training_time=row.get("training_time", ""),
        precompute_time=row.get("precompute_time", ""),
        peak_cpu_ram=row.get("peak_cpu_ram", ""),
        peak_gpu_ram=row.get("peak_gpu_ram", ""),
        cache_bytes=row.get("cache_bytes", cache_bytes),
        predicted_classes=row.get("predicted_classes", ""),
        valid_test_gap=valid_test_gap,
        selected_soft_prior_kl=row.get("selected_soft_prior_kl_to_teacher_prior", row.get("selected_soft_prior_kl", "")),
        notes=(str(row.get("notes", "")) + "; T34 wrapper: teacher probabilities are soft targets only, never input features").strip("; "),
    )
    return apply_t34_promotion_guard(out)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_t34_ratio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("method", "")), _f(row.get("requested_full_node_ratio")))].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method, ratio), group in sorted(groups.items(), key=lambda item: item[0]):
        acc = [_f(row.get("accuracy")) for row in group if row.get("accuracy") not in {"", None}]
        macro = [_f(row.get("macro_f1")) for row in group if row.get("macro_f1") not in {"", None}]
        valid = [_f(row.get("valid_acc")) for row in group if row.get("valid_acc") not in {"", None}]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "requested_full_node_ratio": ratio,
                "seed_count": len(group),
                "accuracy_mean": _mean(acc),
                "accuracy_std": _std(acc),
                "macro_f1_mean": _mean(macro),
                "macro_f1_std": _std(macro),
                "valid_acc_mean": _mean(valid),
                "valid_test_gap_mean": _mean([abs(_f(row.get("valid_acc")) - _f(row.get("accuracy"))) for row in group if row.get("valid_acc") not in {"", None} and row.get("accuracy") not in {"", None}]),
                "promoted_count": sum(1 for row in group if row.get("promotion_status") == "promoted"),
            }
        )
    return out


def build_ratio_curve_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    local = argparse.Namespace(**vars(args))
    local.methods = [_method_to_t33(method) for method in _arg(args, "methods", ["reddit_stt_gamlp_ratio_v2"])]
    t33_rows = build_t33_ratio_rows(local)
    return [t33_to_t34_reddit_row(row, teacher_cache_mode=str(_arg(args, "teacher_cache_mode", "dense_fp16"))) for row in t33_rows]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_ratio_curve_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t34_reddit_stt_ratio_curve.csv"), rows, T34_REQUIRED_FIELDS)
    agg = aggregate_t34_ratio_rows(rows)
    write_csv(_arg(args, "multiseed_csv", "experiments/tables/t34_reddit_stt_multiseed.csv"), agg)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t34_reddit_stt_summary.md"),
        [
            "# T34 Reddit STT",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "valid_acc", "teacher_cache_mode", "promotion_status", "failure_reason"]),
            "",
            "## Multi-Seed Aggregate",
            "",
            *markdown_table(agg, ["method", "requested_full_node_ratio", "seed_count", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "promoted_count"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 Reddit STT ratio curve.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 42])
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
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01])
    parser.add_argument("--methods", nargs="+", default=["reddit_stt_gamlp_ratio_v2", "reddit_stt_sagn_ratio_v2"])
    parser.add_argument("--teacher-cache-mode", default="dense_fp16")
    parser.add_argument("--candidate-nodes", default="all")
    parser.add_argument("--budget-policy", "--stt-budget-policy", dest="budget_policy", default="ratio_adaptive_v2")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.25])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.02])
    parser.add_argument("--lambda-conf", type=float, default=0.0)
    parser.add_argument("--lambda-cover", type=float, default=0.0)
    parser.add_argument("--lambda-calib", type=float, default=0.0)
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
    parser.add_argument("--csv", default="experiments/tables/t34_reddit_stt_ratio_curve.csv")
    parser.add_argument("--multiseed-csv", default="experiments/tables/t34_reddit_stt_multiseed.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_reddit_stt_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
