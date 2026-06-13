from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.products_stt import products_promotion_status
from shadow_hgc.sft.stt_cache import estimate_stt_cache_bytes
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, make_t34_row


PRODUCTS_UCA_REFERENCE = {
    0.0002: (0.6858000868, 0.3094500395, 22),
    0.0004: (0.7000873439, 0.3283601128, 27),
    0.0008: (0.7204511699, 0.3483658099, 27),
    0.0025: (0.7463931668, 0.3791035690, 30),
    0.005: (0.7670750999, 0.3891223435, 31),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def build_teacher_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for teacher in args.teacher_methods:
        rows.append(
            make_t34_row(
                dataset="ogbn-products",
                method=teacher,
                seed=int(args.seed),
                status="blocked",
                failure_reason="missing_products_teacher_cache",
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_teacher_probs=True,
                uses_teacher_logits=True,
                soft_target_only=True,
                teacher_cache_mode="topk8_tail",
                next_action="run scalable products teacher training before STT condensation",
                notes="No local products STT teacher cache was found; UCA/mixup references are not relabeled as STT.",
            )
        )
    return rows


def build_products_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official: list[dict[str, Any]] = []
    balanced: list[dict[str, Any]] = []
    for seed in [int(v) for v in args.seeds]:
        for ratio in [float(v) for v in args.ratios]:
            ref = PRODUCTS_UCA_REFERENCE.get(ratio, ("", "", ""))
            for method in args.methods:
                mode = str(args.teacher_cache_modes[0])
                estimates = estimate_stt_cache_bytes(num_nodes=2_449_029, num_classes=47, mode=mode)
                is_balanced = "balanced" in method
                status, reason = products_promotion_status(method=method, ratio=ratio, accuracy=0.0, macro_f1=0.0, predicted_classes=0)
                del status
                row = make_t34_row(
                    dataset="ogbn-products",
                    method=str(method),
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    status="blocked",
                    failure_reason="missing_products_teacher_cache",
                    promotion_track="sota_chase",
                    promotion_status="not_promoted",
                    teacher_method="products_teacher_ensemble",
                    teacher_cache_mode=mode,
                    teacher_cache_bytes=estimates["teacher_cache_bytes"],
                    teacher_dense_cache_bytes_diagnostic=estimates["teacher_dense_cache_bytes_diagnostic"],
                    cache_compression_ratio=estimates["cache_compression_ratio"],
                    uses_teacher_probs=True,
                    uses_teacher_logits=True,
                    uses_logits_as_input=False,
                    uses_teacher_probs_as_input=False,
                    soft_target_only=True,
                    lambda_hard=0.25,
                    lambda_prior=0.02,
                    lambda_cover=0.10 if is_balanced else 0.01,
                    lambda_calib=0.02,
                    balanced_track=is_balanced,
                    official_track="official" in method,
                    zero_predicted_classes=47,
                    class_coverage_loss="",
                    notes=f"blocked STT row; current non-STT UCA reference at ratio={ratio}: acc={ref[0]}, macro={ref[1]}, predicted_classes={ref[2]}; gate_if_completed={reason}",
                    next_action="python scripts/run_t34_products_stt.py --run-long after products teacher cache exists",
                )
                (balanced if is_balanced else official).append(row)
    return official, balanced


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method", "")), float(row.get("requested_full_node_ratio", 0.0) or 0.0))].append(row)
    out: list[dict[str, Any]] = []
    for (method, ratio), group in sorted(groups.items(), key=lambda item: item[0]):
        acc = [float(row["accuracy"]) for row in group if row.get("accuracy") not in {"", None}]
        macro = [float(row["macro_f1"]) for row in group if row.get("macro_f1") not in {"", None}]
        out.append({"dataset": "ogbn-products", "method": method, "requested_full_node_ratio": ratio, "seed_count": len(group), "accuracy_mean": _mean(acc), "accuracy_std": _std(acc), "macro_f1_mean": _mean(macro), "macro_f1_std": _std(macro), "completed_count": sum(1 for row in group if row.get("status") == "completed_long")})
    return out


def write_outputs(args: argparse.Namespace) -> Path:
    teacher_rows = build_teacher_rows(args)
    official, balanced = build_products_rows(args)
    write_csv(args.teacher_csv, teacher_rows, T34_REQUIRED_FIELDS)
    write_csv(args.official_csv, official, T34_REQUIRED_FIELDS)
    write_csv(args.balanced_csv, balanced, T34_REQUIRED_FIELDS)
    multiseed = aggregate(official + balanced)
    write_csv(args.multiseed_csv, multiseed)
    ensure_report(
        args.report,
        [
            "# T34 Products STT",
            "",
            "Products STT is blocked locally because no scalable products teacher cache is available. Existing UCA/mixup references are kept as references only and are not relabeled as STT.",
            "",
            "## Teacher",
            "",
            *markdown_table(teacher_rows, ["method", "status", "failure_reason", "next_action"]),
            "",
            "## Official",
            "",
            *markdown_table(official[: min(len(official), 12)], ["method", "seed", "requested_full_node_ratio", "status", "failure_reason", "teacher_cache_mode"]),
            "",
            "## Balanced",
            "",
            *markdown_table(balanced[: min(len(balanced), 12)], ["method", "seed", "requested_full_node_ratio", "status", "failure_reason", "teacher_cache_mode", "lambda_cover"]),
        ],
    )
    return Path(args.official_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 Products STT guard/output.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 42])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0002, 0.0004, 0.0008, 0.0025, 0.005])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "products_stt_official",
            "products_stt_balanced",
            "products_stt_dual_head",
            "products_stt_topk8_tail",
            "products_stt_topk16_tail",
            "products_stt_official_plus_uca_init",
            "products_stt_balanced_plus_uca_init",
        ],
    )
    parser.add_argument("--teacher-methods", nargs="+", default=["products_sagn_lite_v5", "products_gamlp_lite_v5", "products_labelreuse_sle_lite", "products_teacher_ensemble"])
    parser.add_argument("--teacher-cache-modes", nargs="+", default=["topk8_tail", "topk16_tail"])
    parser.add_argument("--report-per-class", action="store_true")
    parser.add_argument("--report-resource", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--teacher-csv", default="experiments/tables/t34_products_stt_teacher.csv")
    parser.add_argument("--official-csv", default="experiments/tables/t34_products_stt_official.csv")
    parser.add_argument("--balanced-csv", default="experiments/tables/t34_products_stt_balanced.csv")
    parser.add_argument("--multiseed-csv", default="experiments/tables/t34_products_stt_multiseed.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_products_stt_summary.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
