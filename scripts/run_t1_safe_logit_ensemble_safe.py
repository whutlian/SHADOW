from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from scripts.t1_safe_common import read_csv


FIELDS = ["dataset", "base_variant", "candidate_components", "ensemble_mode", "weights", "valid_acc", "test_acc", "macro_f1", "predicted_class_count", "promotion_status", "promotion_reason", "uses_bounded_edges", "uses_diffusion", "uses_dense_p2"]


def run(args) -> list[dict]:
    rows = []
    correct = [row for row in read_csv(args.correct_table) if row.get("promotion_status") == "promoted"]
    pseudo = [row for row in read_csv(args.pseudo_table) if row.get("promotion_status") == "promoted"]
    datasets = sorted({row["dataset"] for row in correct + pseudo} | {"ogbn-products", "acm"})
    for dataset in datasets:
        components = [row for row in correct + pseudo if row["dataset"] == dataset]
        if len(components) < 2:
            rows.append(
                {
                    "dataset": dataset,
                    "base_variant": components[0]["base_variant"] if components else "",
                    "candidate_components": len(components),
                    "ensemble_mode": "nonnegative_grid",
                    "weights": "",
                    "valid_acc": "",
                    "test_acc": "",
                    "macro_f1": "",
                    "predicted_class_count": "",
                    "promotion_status": "blocked",
                    "promotion_reason": "not_enough_validated_component_logits",
                    "uses_bounded_edges": False,
                    "uses_diffusion": False,
                    "uses_dense_p2": False,
                }
            )
        else:
            best = max(components, key=lambda row: float(row.get("valid_acc_after", 0.0) or 0.0))
            rows.append(
                {
                    "dataset": dataset,
                    "base_variant": best["base_variant"],
                    "candidate_components": len(components),
                    "ensemble_mode": "nonnegative_grid",
                    "weights": "selected_best_component_proxy",
                    "valid_acc": best.get("valid_acc_after", ""),
                    "test_acc": best.get("test_acc_after", ""),
                    "macro_f1": best.get("macro_f1_after", ""),
                    "predicted_class_count": best.get("predicted_class_count_after", ""),
                    "promotion_status": "blocked",
                    "promotion_reason": "ensemble_component_logits_not_persisted",
                    "uses_bounded_edges": False,
                    "uses_diffusion": False,
                    "uses_dense_p2": False,
                }
            )
    write_csv(args.output, rows, FIELDS)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# T1.1 Safe Logit Ensemble Summary\n\n" + f"- CSV: `{args.output}`\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T1.1 safe logit ensemble.")
    parser.add_argument("--correct-table", default="experiments/tables/t1_safe_logit_correct_seed42.csv")
    parser.add_argument("--pseudo-table", default="experiments/tables/t1_pseudo_scap_safe_seed42.csv")
    parser.add_argument("--output", default="experiments/tables/t1_safe_logit_ensemble_safe_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_safe_logit_ensemble_safe_summary.md")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
