from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t32_contract import T32_REQUIRED_FIELDS, make_t32_row, ratio_budget


PRODUCTS_SEED42 = {
    0.0002: (0.6858000868, 0.3094500395, 22),
    0.0004: (0.7000873439, 0.3283601128, 27),
    0.0008: (0.7204511699, 0.3483658099, 27),
    0.0025: (0.7463931668, 0.3791035690, 30),
    0.005: (0.7670750999, 0.3891223435, 31),
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_products_server_command() -> str:
    return (
        "python scripts/run_t32_products_maintenance.py --device cuda --methods products_uca_hybrid_mixup "
        "--ratios 0.0002 0.0004 0.0008 0.0025 0.005 --seeds 1 2 3 4 5 42 "
        "--report-per-class --report-resource --run-long"
    )


def build_products_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", [0.0002, 0.0004, 0.0008, 0.0025, 0.005])]
    seeds = [int(v) for v in _arg(args, "seeds", [42])]
    methods = [str(v) for v in _arg(args, "methods", ["products_uca_hybrid_mixup"])]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            for ratio in ratios:
                budget = ratio_budget("ogbn-products", ratio)
                ref = PRODUCTS_SEED42.get(ratio) if seed == 42 else None
                if ref is None:
                    rows.append(
                        make_t32_row(
                            dataset="ogbn-products",
                            method=method,
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            condensed_nodes=budget,
                            status="blocked",
                            failure_reason="missing_products_seed_reference",
                            promotion_track="safe_main",
                            promotion_status="not_promoted",
                            notes="Products is maintenance-only in T32; non-seed42 rows require replay.",
                            next_action=build_products_server_command(),
                        )
                    )
                    continue
                acc, macro, predicted = ref
                rows.append(
                    make_t32_row(
                        dataset="ogbn-products",
                        method=method,
                        seed=seed,
                        requested_full_node_ratio=ratio,
                        condensed_nodes=budget,
                        accuracy=acc,
                        macro_f1=macro,
                        predicted_classes=predicted,
                        status="carried_forward_reference",
                        failure_reason="products_maintenance_only",
                        promotion_track="safe_main",
                        promotion_status="not_promoted",
                        notes="Carried reference from prior products maintenance table; not promoted as a new T32 result.",
                        next_action=build_products_server_command(),
                    )
                )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_products_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t32_products_maintenance.csv"), rows, T32_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t32_products_maintenance_notes.md"),
        [
            "# T32 Products Maintenance",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_products_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T32 products maintenance rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["products_uca_hybrid_mixup"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0002, 0.0004, 0.0008, 0.0025, 0.005])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-per-class", action="store_true")
    parser.add_argument("--report-resource", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t32_products_maintenance.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_products_maintenance_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
