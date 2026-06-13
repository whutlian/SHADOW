from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.gcrd_gate import compute_gcrd_gate_row


DEFAULT_OURS = [
    "experiments/tables/t34_reddit_stt_ratio_curve.csv",
    "experiments/tables/t34_reddit_stt_cache_ablation.csv",
    "experiments/tables/t34_products_stt_official.csv",
    "experiments/tables/t34_products_stt_balanced.csv",
    "experiments/tables/t34_arxiv_cns_forensic.csv",
    "experiments/tables/t34_arxiv_semantic_stt.csv",
]

FIELDS = [
    "dataset",
    "ratio",
    "ours_method",
    "ours_acc",
    "baseline_acc",
    "gcrd_accuracy_std",
    "absolute_pp_gain",
    "relative_accuracy_gain",
    "relative_error_reduction",
    "passes_5pct_error_reduction",
    "passes_absolute_5pp_if_applicable",
    "teacher_ceiling_gap",
    "ratio_definition_match",
    "mathematically_impossible_under_current_teacher_ceiling",
    "notes",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None, "TODO_EXACT_VALUE"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _todo_baseline_rows() -> list[dict[str, str]]:
    return [
        {"dataset": "Reddit", "ratio_reported": "TODO_EXACT_VALUE", "ratio_definition": "TODO_EXACT_VALUE", "gcrd_accuracy_mean": "TODO_EXACT_VALUE", "gcrd_accuracy_std": "TODO_EXACT_VALUE", "notes": "manual_input_required"},
        {"dataset": "ogbn-products", "ratio_reported": "TODO_EXACT_VALUE", "ratio_definition": "TODO_EXACT_VALUE", "gcrd_accuracy_mean": "TODO_EXACT_VALUE", "gcrd_accuracy_std": "TODO_EXACT_VALUE", "notes": "manual_input_required"},
        {"dataset": "ogbn-arxiv", "ratio_reported": "TODO_EXACT_VALUE", "ratio_definition": "TODO_EXACT_VALUE", "gcrd_accuracy_mean": "TODO_EXACT_VALUE", "gcrd_accuracy_std": "TODO_EXACT_VALUE", "notes": "manual_input_required"},
    ]


def _best_ours(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        if row.get("accuracy") in {"", None}:
            continue
        dataset = str(row.get("dataset", ""))
        ratio = _f(row.get("requested_full_node_ratio", row.get("ratio", 0.0)))
        key = (dataset, ratio)
        if key not in best or _f(row.get("accuracy")) > _f(best[key].get("accuracy")):
            best[key] = row
    return list(best.values())


def build_gate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = _read_csv(args.baseline) or _todo_baseline_rows()
    ours_rows: list[dict[str, str]] = []
    for path in args.ours:
        ours_rows.extend(_read_csv(path))
    best_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _best_ours(ours_rows):
        best_by_dataset[str(row.get("dataset", ""))].append(row)
    out: list[dict[str, Any]] = []
    for base in baseline:
        dataset = str(base.get("dataset", ""))
        candidates = best_by_dataset.get(dataset, [])
        if not candidates:
            candidates = [{"dataset": dataset, "method": "", "requested_full_node_ratio": _f(base.get("ratio_reported")), "accuracy": 0.0, "teacher_accuracy": ""}]
        for ours in sorted(candidates, key=lambda row: _f(row.get("requested_full_node_ratio", 0.0))):
            out.append(
                compute_gcrd_gate_row(
                    dataset=dataset,
                    ratio=_f(ours.get("requested_full_node_ratio", base.get("ratio_reported", 0.0))),
                    ours_method=str(ours.get("method", "")),
                    ours_acc=_f(ours.get("accuracy", 0.0)),
                    baseline_acc=base.get("gcrd_accuracy_mean", "TODO_EXACT_VALUE"),
                    baseline_std=base.get("gcrd_accuracy_std", ""),
                    teacher_ceiling_acc=ours.get("teacher_accuracy", ""),
                    ratio_definition_match=str(base.get("ratio_definition", "")) not in {"", "TODO_EXACT_VALUE"},
                )
            )
    return out


def _split_outputs(rows: list[dict[str, Any]]) -> None:
    mapping = {
        "Reddit": "experiments/tables/t34_reddit_stt_gcrd_gates.csv",
        "ogbn-products": "experiments/tables/t34_products_stt_gcrd_gates.csv",
        "ogbn-arxiv": "experiments/tables/t34_arxiv_gcrd_gates.csv",
    }
    for dataset, path in mapping.items():
        write_csv(path, [row for row in rows if row.get("dataset") == dataset], FIELDS)


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_gate_rows(args)
    csv_path = write_csv(args.out, rows, FIELDS)
    if args.write_split_outputs:
        _split_outputs(rows)
    ensure_report(
        args.report,
        ["# T34 GCRD Error-Reduction Gates", "", *markdown_table(rows, ["dataset", "ratio", "ours_method", "ours_acc", "baseline_acc", "relative_error_reduction", "passes_5pct_error_reduction", "notes"]), "", f"- CSV: `{csv_path}`"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute T34 relative error-reduction gates vs exact GCRD baseline.")
    parser.add_argument("--baseline", default="baselines/gcrd_tpami26_exact.csv")
    parser.add_argument("--ours", nargs="+", default=DEFAULT_OURS)
    parser.add_argument("--out", default="experiments/tables/t34_gcrd_error_reduction_gates.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_gcrd_error_reduction_gates.md")
    parser.add_argument("--write-split-outputs", action="store_true", default=True)
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
