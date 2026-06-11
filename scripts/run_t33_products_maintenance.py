from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS, make_t33_row, ratio_budget
from shadow_hgc.sft.t33_products import aggregate_products_maintenance


PRODUCTS_SEED42 = {
    0.0002: (0.6858000868, 0.3094500395, 22),
    0.0004: (0.7000873439, 0.3283601128, 27),
    0.0008: (0.7204511699, 0.3483658099, 27),
    0.0025: (0.7463931668, 0.3791035690, 30),
    0.005: (0.7670750999, 0.3891223435, 31),
}


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_products_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in [int(v) for v in _arg(args, "seeds", [42])]:
        for method in [str(v) for v in _arg(args, "methods", ["products_uca_hybrid_mixup"])]:
            method_id = "products_uca_hybrid_mixup_multiseed" if method == "products_uca_hybrid_mixup" else method
            for ratio in [float(v) for v in _arg(args, "ratios", [0.0002, 0.0004, 0.0008, 0.0025, 0.005])]:
                ref = PRODUCTS_SEED42.get(ratio) if seed == 42 else None
                if ref is None:
                    rows.append(
                        make_t33_row(
                            dataset="ogbn-products",
                            method=method_id,
                            seed=seed,
                            requested_full_node_ratio=ratio,
                            total_condensed_nodes=ratio_budget("ogbn-products", ratio),
                            status="blocked",
                            failure_reason="missing_products_seed_reference",
                            promotion_track="safe_main",
                            promotion_status="not_promoted",
                            notes="Products maintenance only; non-seed42 replay is required for true multiseed.",
                        )
                    )
                    continue
                acc, macro, pred = ref
                rows.append(
                    make_t33_row(
                        dataset="ogbn-products",
                        method=method_id,
                        seed=seed,
                        requested_full_node_ratio=ratio,
                        total_condensed_nodes=ratio_budget("ogbn-products", ratio),
                        accuracy=acc,
                        macro_f1=macro,
                        predicted_classes=pred,
                        status="carried_forward_reference",
                        failure_reason="products_maintenance_only",
                        promotion_track="safe_main",
                        promotion_status="not_promoted",
                        notes="Carried prior products row; not a new T33 promoted result.",
                    )
                )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_products_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t33_products_maintenance_multiseed.csv"), rows, T33_REQUIRED_FIELDS)
    agg = aggregate_products_maintenance(rows)
    write_csv(_arg(args, "aggregate_csv", "experiments/tables/t33_products_maintenance_aggregate.csv"), agg)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t33_products_maintenance.md"),
        [
            "# T33 Products Maintenance",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            "## Aggregate",
            "",
            *markdown_table(agg, ["method", "requested_full_node_ratio", "seed_count", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 products maintenance multiseed table.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["products_uca_hybrid_mixup"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0002, 0.0004, 0.0008, 0.0025, 0.005])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 42])
    parser.add_argument("--report-per-class", action="store_true")
    parser.add_argument("--report-resource", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_products_maintenance_multiseed.csv")
    parser.add_argument("--aggregate-csv", default="experiments/tables/t33_products_maintenance_aggregate.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_products_maintenance.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
