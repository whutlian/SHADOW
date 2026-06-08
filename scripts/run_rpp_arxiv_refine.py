from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment
from scripts.run_rpp_common import base_row, write_csv, write_report


def _config(feature_variant: str, model_variant: str) -> dict:
    config = {
        "method_name": "Shadow-HGC-R++",
        "feature_mode": "diffusion",
        "diffusion_steps": (1, 2),
        "include_highpass": True,
        "skeleton_policy": "coverage",
        "skeleton_coverage": 0.65,
        "skeleton_k_max": 8,
        "projection_type": "random",
    }
    if feature_variant == "diffusion_X0X1X2_highpass_blocknorm":
        config["block_norm"] = "standardize"
    elif feature_variant != "diffusion_X0X1X2_highpass":
        raise ValueError(f"unknown feature variant: {feature_variant}")
    if model_variant == "relation_linear_no_final_relu":
        config["model_type"] = "relation_linear"
        config["relation_gate"] = True
    elif model_variant == "shadow_fusion":
        config["model_type"] = "shadow_fusion"
        config["relation_gate"] = True
        config["inference_dst_chunk_size"] = 8192
    else:
        raise ValueError(f"unknown model variant: {model_variant}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R++ ogbn-arxiv refinement, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/arxiv_rpp_refine_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/arxiv_rpp_refine_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/arxiv_rpp_refine_seed42")
    args = parser.parse_args()

    dataset = "ogbn-arxiv"
    try:
        graph = load_ogb_node_property_dataset(dataset, download=args.download)
    except Exception as exc:
        path = Path(args.log_dir) / f"{dataset}_load_failed.json"
        payload = {"dataset": dataset, "seed": args.seed, "status": "data_not_available", "reason": str(exc)}
        write_json_summary(path, payload)
        rows = [base_row(path, dataset=dataset, variant="load_failed", summary=payload)]
        write_csv(args.output, rows)
        write_report(args.report_output, title="ogbn-arxiv R++ Refinement Summary", rows=rows, csv_path=args.output, previous_best={dataset: 0.5369})
        return

    ratios = [0.02, 0.06, 0.12]
    feature_variants = ["diffusion_X0X1X2_highpass", "diffusion_X0X1X2_highpass_blocknorm"]
    model_variants = ["relation_linear_no_final_relu", "shadow_fusion"]
    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for feature_variant in feature_variants:
        for model_variant in model_variants:
            for ratio in ratios:
                variant = f"{feature_variant}_{model_variant}"
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
                            feature_dim=128,
                            loss_type="sqrt_weighted_logit_adjusted",
                            min_proto_per_class=4,
                            budget_alpha=0.5,
                            multiscale_dim=128,
                            assignment_chunk_size=4096,
                            **_config(feature_variant, model_variant),
                        )
                    rows.append(base_row(path, dataset=dataset, variant=variant, summary=summary))
                except Exception as exc:
                    payload = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": args.seed,
                        "ratio": ratio,
                        "loss_type": "sqrt_weighted_logit_adjusted",
                        "status": exception_status(exc),
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    write_json_summary(path, payload)
                    rows.append(base_row(path, dataset=dataset, variant=variant, summary=payload))
    write_csv(args.output, rows)
    write_report(args.report_output, title="ogbn-arxiv R++ Refinement Summary", rows=rows, csv_path=args.output, previous_best={dataset: 0.5369})
    print(f"wrote {args.output}")
    print(f"wrote {args.report_output}")


if __name__ == "__main__":
    main()
