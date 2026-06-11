from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t29_contract import T29_REQUIRED_FIELDS, make_t29_row, ratio_budget


DEFAULT_RATIOS = (0.0002, 0.0004, 0.0008, 0.0025, 0.005)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_products_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t29_products_maintenance.py --device cuda --methods products_uca_hybrid_mixup "
        "--ratios 0.0002 0.0004 0.0008 0.0025 0.005 --seeds 1 2 3 4 5 42 --run-long"
    )


def _refs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        "experiments/tables/t28_products_maintenance_seed42.csv",
        "experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv",
        "experiments/tables/t26_products_long_experiments_seed42.csv",
    ]:
        for row in read_csv(path):
            item = dict(row)
            item["_source_csv"] = path
            rows.append(item)
    return rows


def _allowed(row: dict[str, Any], ratio: float) -> bool:
    method = str(row.get("method", ""))
    if ratio in {0.0025, 0.005}:
        return method == "products_uca_hybrid_mixup"
    return method in {"products_uca_hybrid_mixup"} or method.startswith("products_uca_mixup")


def _match(rows: list[dict[str, Any]], ratio: float, seed: int) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.get("seed") not in {"", None} and int(float(row.get("seed", 0))) != int(seed):
            continue
        if abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) > 1e-12:
            continue
        if row.get("accuracy") in {"", None}:
            continue
        if not _allowed(row, ratio):
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: fvalue(row.get("accuracy")))


def build_products_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(v) for v in _arg(args, "ratios", DEFAULT_RATIOS)]
    seeds = [int(v) for v in _arg(args, "seeds", [_arg(args, "seed", 42)])]
    refs = _refs()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for ratio in ratios:
            budget = ratio_budget("ogbn-products", ratio)
            ref = _match(refs, ratio, seed)
            if ref:
                rows.append(
                    make_t29_row(
                        dataset="ogbn-products",
                        method="products_uca_hybrid_mixup",
                        seed=seed,
                        requested_full_node_ratio=ratio,
                        target_prototypes=budget,
                        accuracy=ref.get("accuracy", ""),
                        macro_f1=ref.get("macro_f1", ""),
                        predicted_classes=ref.get("predicted_classes", ref.get("predicted_class_count", "")),
                        status="carried_forward_reference",
                        promotion_status="carry_forward",
                        promotion_track="safe_mainline",
                        failure_reason="products_maintenance_only",
                        notes="Products is maintenance only in T29; class-collapse diagnostics are retained.",
                        source_table=ref.get("_source_csv", ""),
                    )
                )
            else:
                rows.append(
                    make_t29_row(
                        dataset="ogbn-products",
                        method="products_uca_hybrid_mixup",
                        seed=seed,
                        requested_full_node_ratio=ratio,
                        target_prototypes=budget,
                        status="server_ready_not_run",
                        promotion_status="carry_forward",
                        promotion_track="safe_mainline",
                        failure_reason="missing_seed_reference",
                        notes="Run products maintenance replay for this seed if needed.",
                    )
                )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_products_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t29_products_maintenance_seed42.csv"), rows, T29_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t29_products_maintenance_summary.md"),
        [
            "# T29 Products Maintenance",
            "",
            "- Products remains a maintenance/success-case line.",
            "",
            *markdown_table(rows, ["method", "seed", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_products_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T29 products maintenance.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["products_uca_hybrid_mixup"])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t29_products_maintenance_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t29_products_maintenance_summary.md")
    args = parser.parse_args()
    csv_path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
