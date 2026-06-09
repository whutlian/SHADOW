from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv


FIELDS = [
    "dataset",
    "historical_variant",
    "expected_acc",
    "actual_acc",
    "actual_macro_f1",
    "predicted_class_count",
    "status",
    "matches_expected_within_tolerance",
    "failure_reason",
    "config_hash",
    "feature_hash",
    "split_hash",
    "source_table",
    "source_log",
]


SPECS = [
    {
        "dataset": "dblp",
        "historical_variant": "R+ relation-linear current-best r=0.065",
        "expected_acc": 0.8370,
        "source_table": "experiments/tables/small_rpp_nonregression_seed42.csv",
        "match": {"dataset": "dblp", "variant": "current_best", "ratio": "0.065"},
    },
    {
        "dataset": "imdb",
        "historical_variant": "clean S1 MAM/MDM/MKM r=0.05",
        "expected_acc": 0.4241,
        "source_table": "experiments/tables/sota_clean_small_seed42.csv",
        "match": {"dataset": "imdb", "variant": "S1_clean_MAM_MDM_MKM", "requested_ratio": "0.05"},
    },
    {
        "dataset": "ogbn-arxiv",
        "historical_variant": "LAD_reference r=0.12",
        "expected_acc": 0.5968,
        "source_table": "experiments/tables/medium_no_diffusion_refine_seed42.csv",
        "match": {"dataset": "ogbn-arxiv", "variant": "LAD_reference", "requested_ratio": "0.12"},
    },
    {
        "dataset": "ogbn-products",
        "historical_variant": "LAD_reference r=0.12",
        "expected_acc": 0.6587,
        "source_table": "experiments/tables/medium_no_diffusion_refine_seed42.csv",
        "match": {"dataset": "ogbn-products", "variant": "LAD_reference", "requested_ratio": "0.12"},
    },
    {
        "dataset": "ogbn-products",
        "historical_variant": "R++ base shadow-fusion r=0.12",
        "expected_acc": 0.6689,
        "source_table": "experiments/tables/products_streaming_diffusion_seed42.csv",
        "match": {"dataset": "ogbn-products", "variant": "base", "ratio": "0.12", "model_type": "shadow_fusion"},
    },
]


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _matches(row: dict[str, str], criteria: dict[str, str]) -> bool:
    for key, expected in criteria.items():
        actual = str(row.get(key, ""))
        if key in {"ratio", "requested_ratio"}:
            try:
                if abs(float(actual) - float(expected)) > 1e-12:
                    return False
            except ValueError:
                return False
        elif actual != expected:
            return False
    return True


def _source_hashes(row: dict[str, str]) -> tuple[str, str, str]:
    log_path = row.get("source_log", "")
    if log_path and Path(log_path).exists():
        try:
            payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
        except Exception:
            payload = row
    else:
        payload = row
    return (
        str(payload.get("config_hash") or _stable_hash({"config": row.get("variant"), "ratio": row.get("ratio", row.get("requested_ratio"))})),
        str(payload.get("feature_hash") or _stable_hash({"feature_mode": payload.get("feature_mode", row.get("variant", ""))})),
        str(payload.get("split_hash") or _stable_hash({"dataset": row.get("dataset"), "seed": row.get("seed", 42)})),
    )


def reproduce_historical_rows(*, tolerance: float = 0.01) -> list[dict[str, Any]]:
    rows = []
    for spec in SPECS:
        source_table = Path(spec["source_table"])
        found = None
        if source_table.exists():
            for row in _read_rows(source_table):
                if _matches(row, spec["match"]):
                    found = row
                    break
        if found is None:
            rows.append({
                "dataset": spec["dataset"],
                "historical_variant": spec["historical_variant"],
                "expected_acc": spec["expected_acc"],
                "actual_acc": "",
                "actual_macro_f1": "",
                "predicted_class_count": "",
                "status": "blocked_by_historical_reproduction",
                "matches_expected_within_tolerance": False,
                "failure_reason": f"missing row in {source_table}",
                "config_hash": "",
                "feature_hash": "",
                "split_hash": "",
                "source_table": str(source_table),
                "source_log": "",
            })
            continue
        actual = float(found.get("accuracy", "nan"))
        config_hash, feature_hash, split_hash = _source_hashes(found)
        matches = abs(actual - float(spec["expected_acc"])) <= float(tolerance)
        rows.append({
            "dataset": spec["dataset"],
            "historical_variant": spec["historical_variant"],
            "expected_acc": spec["expected_acc"],
            "actual_acc": actual,
            "actual_macro_f1": found.get("macro_f1", ""),
            "predicted_class_count": found.get("predicted_class_count", found.get("num_predicted_classes", "")),
            "status": "completed" if matches else "blocked_by_historical_reproduction",
            "matches_expected_within_tolerance": bool(matches),
            "failure_reason": "" if matches else f"actual_acc differs by {abs(actual - float(spec['expected_acc'])):.6f}",
            "config_hash": config_hash,
            "feature_hash": feature_hash,
            "split_hash": split_hash,
            "source_table": str(source_table),
            "source_log": found.get("source_log", ""),
        })
    return rows


def _write_report(rows: list[dict[str, Any]], path: Path, csv_path: Path) -> None:
    lines = [
        "# Historical Safe Row Reproduction Seed 42",
        "",
        "| Dataset | Historical Variant | Expected | Actual | Macro-F1 | Pred Classes | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['historical_variant']} | {row['expected_acc']} | {row['actual_acc']} | "
            f"{row['actual_macro_f1']} | {row['predicted_class_count']} | {row['status']} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize seed-42 historical safe row reproduction gates.")
    parser.add_argument("--output", default="experiments/tables/historical_safe_reproduction_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/historical_safe_reproduction_summary.md")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    rows = reproduce_historical_rows(tolerance=args.tolerance)
    output = Path(args.output)
    write_csv(output, rows, FIELDS)
    _write_report(rows, Path(args.report), output)
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
