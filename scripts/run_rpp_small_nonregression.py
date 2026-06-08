from __future__ import annotations

import argparse
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
from scripts.run_rpp_common import base_row, write_csv, write_report


def _config(variant: str) -> dict:
    if variant == "current_best":
        return {
            "method_name": "Shadow-HGC-R+",
            "feature_mode": "metapath",
            "metapath_signature": True,
            "metapath_model_input": True,
            "shadow_policy": "rank_adaptive",
            "adaptive_b": True,
            "relation_gate": True,
            "model_type": "relation_linear",
        }
    if variant == "shadow_fusion_blocknorm":
        return {
            "method_name": "Shadow-HGC-R++",
            "feature_mode": "metapath",
            "metapath_signature": True,
            "metapath_model_input": True,
            "shadow_policy": "rank_adaptive",
            "adaptive_b": True,
            "relation_gate": True,
            "model_type": "shadow_fusion",
            "block_norm": "standardize",
            "assignment_chunk_size": 2048,
            "inference_dst_chunk_size": 4096,
        }
    raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R++ ACM/DBLP non-regression checks, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/small_rpp_nonregression_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/small_rpp_nonregression_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/small_rpp_nonregression_seed42")
    args = parser.parse_args()

    specs = [("acm", 0.096), ("dblp", 0.005), ("dblp", 0.065)]
    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ratio in specs:
        graph = load_processed_small_dataset(dataset)
        for variant in ["current_best", "shadow_fusion_blocknorm"]:
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
                        k_s=2,
                        min_proto_per_class=4,
                        budget_alpha=0.5,
                        **_config(variant),
                    )
                rows.append(base_row(path, dataset=dataset, variant=variant, summary=summary))
            except Exception as exc:
                payload = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": args.seed,
                    "ratio": ratio,
                    "loss_type": "clipped",
                    "status": exception_status(exc),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_summary(path, payload)
                rows.append(base_row(path, dataset=dataset, variant=variant, summary=payload))
    write_csv(args.output, rows)
    write_report(args.report_output, title="Small R++ Non-Regression Summary", rows=rows, csv_path=args.output, previous_best={"acm": 0.8432, "dblp": 0.8370})
    print(f"wrote {args.output}")
    print(f"wrote {args.report_output}")


if __name__ == "__main__":
    main()
