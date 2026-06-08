from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _config(variant: str) -> dict:
    if variant == "base":
        return {"method_name": "Shadow-HGC-R-1", "feature_mode": "base", "shadow_policy": "fixed", "adaptive_b": False, "relation_gate": False}
    return {
        "method_name": "Shadow-HGC-R+",
        "feature_mode": "metapath",
        "metapath_signature": True,
        "metapath_model_input": True,
        "shadow_policy": "rank_adaptive",
        "adaptive_b": True,
        "b_max": 4,
        "relation_gate": True,
        "relation_gate_init": 1.0,
    }


def _row(path: Path, dataset: str, variant: str, summary: dict) -> dict:
    rank = summary.get("diagnostics", {}).get("rank", {})
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": summary.get("seed", ""),
        "ratio": summary.get("ratio", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "shadow_recon_err_by_relation": json.dumps(summary.get("shadow_recon_err_by_relation", {}), sort_keys=True),
        "effective_rank_by_relation": json.dumps(
            {
                relation: max(float(values.get("stable_rank", 0.0)), float(values.get("entropy_effective_rank", 0.0)))
                for relation, values in rank.items()
            },
            sort_keys=True,
        ),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "status": summary.get("status", "completed"),
        "source_log": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACM/DBLP R+ regression checks with seed 42.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/acm_dblp_rplus_regression_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/acm_dblp_rplus_regression_seed42")
    args = parser.parse_args()

    specs = [("acm", 0.096), ("dblp", 0.005), ("dblp", 0.065)]
    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ratio in specs:
        graph = load_processed_small_dataset(dataset)
        for variant in ["base", "rplus"]:
            path = log_dir / f"{dataset}_{variant}_{ratio_slug(ratio)}_seed{args.seed}.json"
            try:
                if args.skip_existing and path.exists():
                    summary = json.loads(path.read_text(encoding="utf-8"))
                else:
                    summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=path,
                        seed=args.seed,
                        epochs=args.epochs,
                        budget_mode="ratio",
                        ratio=ratio,
                        ratio_base="train_target",
                        feature_dim=64,
                        projection_type="raw",
                        loss_type="clipped",
                        model_type="relation_linear",
                        k_s=2,
                        min_proto_per_class=4,
                        budget_alpha=0.5,
                        **_config(variant),
                    )
                rows.append(_row(path, dataset, variant, summary))
            except Exception as exc:
                payload = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": args.seed,
                    "ratio": ratio,
                    "status": exception_status(exc),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_summary(path, payload)
                rows.append(_row(path, dataset, variant, payload))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
