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
    base = {
        "method_name": "Shadow-HGC-R++" if variant != "full_rplus_current" else "Shadow-HGC-R+",
        "feature_mode": "metapath",
        "metapath_signature": True,
        "metapath_model_input": True,
        "projection_type": "raw",
        "model_type": "relation_linear",
        "shadow_policy": "rank_adaptive",
        "adaptive_b": True,
        "b_max": 4,
        "relation_gate": True,
        "relation_gate_init": 1.0,
    }
    if variant == "full_rplus_current":
        return base
    if variant == "full_rplus_blocknorm":
        base.update({"block_norm": "standardize"})
        return base
    if variant == "full_rplus_shadow_fusion":
        base.update({"model_type": "shadow_fusion", "block_norm": "standardize", "inference_dst_chunk_size": 2048})
        return base
    if variant == "full_rplus_shadow_fusion_adaptive_b":
        base.update(
            {
                "model_type": "shadow_fusion",
                "block_norm": "standardize",
                "assignment_chunk_size": 2048,
                "inference_dst_chunk_size": 2048,
                "ratio_mode": "total_nodes",
            }
        )
        return base
    raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R++ IMDB rescue v2, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/imdb_rpp_rescue_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/imdb_rpp_rescue_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/imdb_rpp_rescue_seed42")
    args = parser.parse_args()

    graph = load_processed_small_dataset("imdb")
    variants = ["full_rplus_current", "full_rplus_blocknorm", "full_rplus_shadow_fusion", "full_rplus_shadow_fusion_adaptive_b"]
    losses = ["clipped", "class_balanced"]
    ratios = [0.005, 0.025, 0.05]
    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for ratio in ratios:
            for loss_type in losses:
                path = log_dir / f"imdb_{variant}_{loss_type}_{ratio_slug(ratio)}_seed{args.seed}.json"
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
                            loss_type=loss_type,
                            k_s=2,
                            min_proto_per_class=4,
                            budget_alpha=0.5,
                            shadow_min_per_relation=8,
                            shadow_max_multiplier=2.0,
                            **_config(variant),
                        )
                    rows.append(base_row(path, dataset="imdb", variant=variant, summary=summary))
                except Exception as exc:
                    payload = {
                        "dataset": "imdb",
                        "variant": variant,
                        "seed": args.seed,
                        "ratio": ratio,
                        "loss_type": loss_type,
                        "status": exception_status(exc),
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    write_json_summary(path, payload)
                    rows.append(base_row(path, dataset="imdb", variant=variant, summary=payload))
    write_csv(args.output, rows)
    write_report(
        args.report_output,
        title="IMDB R++ Rescue Summary",
        rows=rows,
        csv_path=args.output,
        previous_best={"imdb": 0.3810},
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report_output}")


if __name__ == "__main__":
    main()
