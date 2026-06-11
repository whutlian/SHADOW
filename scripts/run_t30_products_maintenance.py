from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, make_t30_row, ratio_budget


PRODUCTS_SEED42 = {
    0.0002: (0.6858000868, 0.3094500395, 22),
    0.0004: (0.7000873439, 0.3283601128, 27),
    0.0008: (0.7204511699, 0.3483658099, 27),
    0.0025: (0.7463931668, 0.3791035690, 30),
    0.005: (0.7670750999, 0.3891223435, 31),
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_products_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t30_products_maintenance.py --device cuda --methods products_uca_hybrid_mixup "
        "--ratios 0.0002 0.0004 0.0008 0.0025 0.005 --seeds 1 2 3 4 5 42 "
        "--report-class-histograms --run-long"
    )


def build_products_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", [0.0002, 0.0004, 0.0008, 0.0025, 0.005])]
    seeds = [int(v) for v in _arg(args, "seeds", [_arg(args, "seed", 42)])]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for ratio in ratios:
            budget = ratio_budget("ogbn-products", ratio)
            ref = PRODUCTS_SEED42.get(float(ratio)) if seed == 42 else None
            if ref is None:
                rows.append(
                    make_t30_row(
                        dataset="ogbn-products",
                        method="products_uca_hybrid_mixup",
                        seed=seed,
                        requested_full_node_ratio=ratio,
                        num_codewords=budget,
                        status="blocked",
                        promotion_track="safe_main",
                        failure_reason="missing_products_seed_reference",
                        notes="Products is maintenance only; this seed needs the server maintenance replay.",
                        next_action=build_products_server_command(seed),
                        transfer_eval_type="maintenance_reference",
                    )
                )
                continue
            acc, macro, predicted = ref
            rows.append(
                make_t30_row(
                    dataset="ogbn-products",
                    method="products_uca_hybrid_mixup",
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    num_codewords=budget,
                    accuracy=acc,
                    macro_f1=macro,
                    predicted_classes=predicted,
                    status="carried_forward_reference",
                    promotion_status="not_promoted",
                    promotion_track="safe_main",
                    failure_reason="products_maintenance_only",
                    notes="T30 products maintenance carries forward the current UCA+mixup reference without new promotion.",
                    transfer_eval_type="maintenance_reference",
                    extra={"class_histogram_json": json.dumps({"predicted_classes": predicted}, sort_keys=True)},
                )
            )
    return rows


def _summary_stats(rows: list[dict[str, Any]]) -> list[str]:
    values = [float(row["accuracy"]) for row in rows if row.get("accuracy") not in {"", None}]
    if not values:
        return ["- No completed products maintenance metrics were available."]
    return [
        f"- Accuracy mean/std over available rows: {statistics.mean(values):.6f} / {(statistics.pstdev(values) if len(values) > 1 else 0.0):.6f}",
    ]


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_products_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t30_products_maintenance.csv"), rows, T30_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t30_products_maintenance_notes.md"),
        [
            "# T30 Products Maintenance",
            "",
            "- Products remains maintenance-only in T30.",
            *_summary_stats(rows),
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_products_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T30 products maintenance.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["products_uca_hybrid_mixup"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0002, 0.0004, 0.0008, 0.0025, 0.005])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-class-histograms", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t30_products_maintenance.csv")
    parser.add_argument("--report", default="experiments/summaries/t30_products_maintenance_notes.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
