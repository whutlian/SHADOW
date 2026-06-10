from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row


REDDIT_NODES = 232_965
REDDIT_T24_REFERENCE = {0.005: {"accuracy": 0.9244564924689873, "macro_f1": 0.8862562817528249}}
METHODS = [
    "reddit_current_sft_signature_random",
    "reddit_current_sft_signature_medoid",
    "reddit_current_sft_signature_kcenter",
    "reddit_sft_hnr_fdm_hybrid",
    "reddit_tuned_balanced_trainer",
    "reddit_sft_signature_mixup",
    "reddit_true_shadow_b1",
]

FIELDS = T26_REQUIRED_FIELDS + ["source_t25_table", "seed_sweep_mean_acc", "seed_sweep_std_acc", "seed_sweep_mean_macro_f1", "seed_sweep_std_macro_f1", "validation_selected"]


def _source_row(rows: list[dict[str, str]], *, ratio: float, method: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row.get("dataset") == "Reddit"
        and abs(float(row.get("requested_full_node_ratio", 0.0)) - float(ratio)) < 1e-12
        and row.get("method") in {method, method.replace("reddit_", "")}
    ]
    if not candidates:
        return None
    return candidates[0]


def _metric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            out.append(float(value))
    return out


def build_rows(seed: int = 42, t25_csv: str | Path = "experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv") -> list[dict[str, Any]]:
    source_rows = read_csv(t25_csv)
    out: list[dict[str, Any]] = []
    ratios = [0.005, 0.01]
    seeds = [1, 2, 3, 4, 5]
    for ratio in ratios:
        total = max(1, int(round(REDDIT_NODES * ratio)))
        for method in METHODS:
            method_rows: list[dict[str, Any]] = []
            for sweep_seed in seeds:
                source = _source_row(source_rows, ratio=ratio, method=method)
                use_source = source is not None and int(float(source.get("seed", -1))) == int(sweep_seed)
                shadow = "true_shadow" in method
                failure = "seed_not_run"
                status = "ready_not_run"
                acc: float | str = ""
                macro: float | str = ""
                pred: int | str = ""
                if use_source:
                    acc = float(source.get("accuracy", ""))
                    macro = float(source.get("macro_f1", ""))
                    pred = int(float(source.get("predicted_class_count", source.get("predicted_classes", "0"))))
                    status = "completed_reuse_existing_t25_seed"
                    failure = "t26_trainer_recipe_not_rerun"
                if shadow:
                    failure = "true_shadow_graph_not_materialized"
                    status = "diagnostic_shadow_not_trained"
                row = make_t26_row(
                    dataset="Reddit",
                    method=method,
                    requested_full_node_ratio=ratio,
                    original_total_nodes=REDDIT_NODES,
                    target_prototypes=total,
                    shadow_nodes=0,
                    total_condensed_edges=total,
                    seed=int(sweep_seed),
                    accuracy=acc,
                    macro_f1=macro,
                    predicted_classes=pred,
                    status=status,
                    promotion_status="not_promoted",
                    promotion_reason=failure,
                    failure_reason=failure,
                    notes="T26 Reddit table declares the required seed/trainer/mixup grid; rows are promoted only after actual seed sweep results meet no-regression and target gates.",
                    trainer_recipe="balanced_adamw_label_smoothing_mixup" if "balanced" in method or "mixup" in method else "standard_adamw",
                    trainer_balanced_batches="balanced" in method,
                    trainer_label_smoothing=0.05 if "balanced" in method else 0.0,
                    trainer_mixup_alpha=0.4 if "mixup" in method else 0.0,
                    shadow_graph_materialized=False,
                    shadow_b=1 if shadow else "",
                    source_t25_table=str(t25_csv),
                    validation_selected=False,
                )
                ref = REDDIT_T24_REFERENCE.get(float(ratio))
                if acc != "" and ref is not None and (float(acc) < ref["accuracy"] or float(macro) < ref["macro_f1"]):
                    row["failure_reason"] = "no_regression_gate_not_met"
                    row["promotion_reason"] = "no_regression_gate_not_met"
                method_rows.append(row)
                out.append(row)
            accs = _metric_values(method_rows, "accuracy")
            macros = _metric_values(method_rows, "macro_f1")
            completed = [row for row in method_rows if row.get("accuracy") not in {"", None}]
            if len(completed) == len(seeds) and accs:
                for row in method_rows:
                    row["seed_sweep_mean_acc"] = statistics.mean(accs)
                    row["seed_sweep_std_acc"] = statistics.pstdev(accs) if len(accs) > 1 else 0.0
                    row["seed_sweep_mean_macro_f1"] = statistics.mean(macros)
                    row["seed_sweep_std_macro_f1"] = statistics.pstdev(macros) if len(macros) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T26 Reddit seed/trainer/mixup sweep table.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t25-csv", default="experiments/tables/t25_reddit_hnr_fdm_ratio_sweep_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t26_reddit_seed_trainer_mixup_sweep.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_reddit_trainer_mixup_notes.md")
    args = parser.parse_args()
    rows = build_rows(seed=int(args.seed), t25_csv=args.t25_csv)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Reddit Trainer Mixup Notes",
            "",
            "- Required seeds 1..5 and ratios 0.50%/1.00% are declared.",
            "- Missing seed runs are marked ready_not_run; no seed42 replay is promoted as a seed sweep.",
            "- True shadow rows remain diagnostic until a schema-preserving shadow graph is materialized and trained.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "seed", "method", "status", "accuracy", "macro_f1", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
