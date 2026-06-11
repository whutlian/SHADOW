from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t28_contract import PRODUCTS_MAINTENANCE_FIELDS, make_products_maintenance_row


DEFAULT_RATIOS: tuple[float, ...] = (0.0002, 0.0004, 0.0008, 0.0025, 0.005)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_products_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t28_products_maintenance.py --device cuda "
        "--methods products_uca_hybrid_mixup "
        "--ratios 0.0002 0.0004 0.0008 0.0025 0.005 "
        f"--seed {int(seed)}"
    )


def _load_references(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        _arg(args, "t27_tiny_csv", "experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv"),
        _arg(args, "t26_long_csv", "experiments/tables/t26_products_long_experiments_seed42.csv"),
    ]:
        for row in read_csv(path):
            copied = dict(row)
            copied["_source_csv"] = str(path)
            rows.append(copied)
    return rows


def _allowed_uca_reference(row: dict[str, Any], ratio: float) -> bool:
    method = str(row.get("method", ""))
    if float(ratio) in {0.0025, 0.005}:
        return method == "products_uca_hybrid_mixup"
    return method.startswith("products_uca_mixup")


def _best_ratio_reference(rows: list[dict[str, Any]], ratio: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if abs(fvalue(row.get("requested_full_node_ratio")) - float(ratio)) < 1e-12
        and row.get("accuracy") not in {"", None}
        and _allowed_uca_reference(row, ratio)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: fvalue(row.get("accuracy")))


def build_products_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(value) for value in _arg(args, "ratios", DEFAULT_RATIOS)]
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    references = _load_references(args)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        ref = _best_ratio_reference(references, ratio)
        if ref is None:
            rows.append(
                make_products_maintenance_row(
                    method="products_uca_hybrid_mixup",
                    seed=seed,
                    requested_full_node_ratio=ratio,
                    status="completed_smoke" if smoke else "blocked_missing_reference",
                    promotion_status="carry_forward",
                    failure_reason_if_not_promoted="missing_products_reference_csv",
                    notes="Products maintenance row without local replay reference.",
                )
            )
            continue
        predicted = ref.get("predicted_classes", ref.get("predicted_class_count", ""))
        rows.append(
            make_products_maintenance_row(
                method="products_uca_hybrid_mixup",
                seed=seed,
                requested_full_node_ratio=ratio,
                accuracy=ref.get("accuracy", ""),
                macro_f1=ref.get("macro_f1", ""),
                predicted_classes=predicted,
                status="carried_forward_reference",
                promotion_status="carry_forward",
                failure_reason_if_not_promoted="products_maintenance_only_not_new_promotion",
                notes=f"Best local products reference for this ratio imported from {ref.get('_source_csv', '')}.",
                extra={
                    "predicted_hist_json": ref.get("predicted_class_histogram_json", ref.get("predicted_class_counts_json", "")),
                    "selected_class_hist_json": ref.get("synthetic_class_histogram_json", ref.get("selected_class_counts_json", "")),
                    "official_accuracy_track": "retained",
                    "balanced_robustness_track": "reported_predicted_classes_macro_f1",
                },
            )
        )
    return rows


def write_products_outputs(args: argparse.Namespace) -> Path:
    rows = build_products_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t28_products_maintenance_seed42.csv"), rows, PRODUCTS_MAINTENANCE_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t28_products_maintenance_summary.md"),
        [
            "# T28 Products Maintenance",
            "",
            "- Products is frozen as a strong maintenance line in T28.",
            "- Accuracy, macro-F1, and predicted-class count are retained so class collapse is visible.",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "predicted_classes", "status", "failure_reason_if_not_promoted"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_products_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T28 products maintenance table.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["products_uca_hybrid_mixup"])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--t27-tiny-csv", default="experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv")
    parser.add_argument("--t26-long-csv", default="experiments/tables/t26_products_long_experiments_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t28_products_maintenance_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_products_maintenance_summary.md")
    args = parser.parse_args()
    csv_path = write_products_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
