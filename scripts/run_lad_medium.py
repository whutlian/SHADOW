from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import (
    DATASET_LOSS,
    LAD_TABLE_FIELDS,
    LAD_VARIANTS,
    MEDIUM_RATIOS,
    lad_feature_mode,
    ratio_label,
    summary_to_lad_row,
    write_csv,
)
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-L LAD medium matrix.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="experiments/tables/lad_stage_medium_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/lad_stage_medium_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compiled-hidden-dim", type=int, default=256)
    parser.add_argument("--compiled-dropout", type=float, default=0.3)
    parser.add_argument("--compiled-block-gate", type=_bool_arg, default=True)
    args = parser.parse_args()

    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            for ratio in MEDIUM_RATIOS[dataset]:
                for variant in LAD_VARIANTS:
                    log_path = log_dir / f"{dataset}_{variant.name}_r{ratio_label(ratio)}_seed{args.seed}.json"
                    summary = {
                        "dataset": dataset,
                        "variant": variant.name,
                        "seed": args.seed,
                        "ratio": ratio,
                        "status": "ogb_data_not_available",
                        "reason": str(exc),
                        "compiled_head": variant.compiled_head,
                        "label_affinity": variant.label_affinity,
                        "boundary_prototypes": variant.boundary_prototypes,
                        "diffusion_enabled": False,
                        "diffusion_status": "diagnostic_only",
                    }
                    write_json_summary(log_path, summary)
                    rows.append(summary_to_lad_row(summary, dataset=dataset, ratio=ratio, variant=variant.name, log_path=log_path))
            continue
        for ratio in MEDIUM_RATIOS[dataset]:
            for variant in LAD_VARIANTS:
                log_path = log_dir / f"{dataset}_{variant.name}_r{ratio_label(ratio)}_seed{args.seed}.json"
                if args.skip_existing and log_path.exists():
                    summary = json.loads(log_path.read_text(encoding="utf-8"))
                else:
                    try:
                        summary = run_shadow_hgc_experiment(
                            graph,
                            output_path=log_path,
                            method_name="Shadow-HGC-L" if variant.compiled_head else "Shadow-HGC-R-1",
                            stage="lad",
                            seed=args.seed,
                            epochs=args.epochs,
                            budget_mode="ratio",
                            ratio=ratio,
                            ratio_base="train_target",
                            budget_rounding="nearest",
                            feature_dim=128,
                            projection_type="random",
                            model_type="shadow_fusion",
                            hidden_dim=128,
                            dropout=0.3,
                            lr=0.03,
                            weight_decay=1e-4,
                            loss_type=DATASET_LOSS[dataset],
                            logit_adjustment_tau=1.0,
                            feature_mode=lad_feature_mode(label_affinity=variant.label_affinity, metapath=False),
                            diffusion_enabled=False,
                            diffusion_status="diagnostic_only",
                            label_affinity=variant.label_affinity,
                            label_affinity_mode="target_target",
                            label_affinity_self_exclude=True,
                            label_affinity_block_norm="row_l1",
                            compiled_head=variant.compiled_head,
                            compiled_head_fusion="concat_mlp",
                            compiled_hidden_dim=args.compiled_hidden_dim,
                            compiled_dropout=args.compiled_dropout,
                            compiled_block_gate=args.compiled_block_gate,
                            compiled_demand_source="shadow_reconstructed",
                            boundary_prototypes=variant.boundary_prototypes,
                            boundary_fraction=0.3,
                            boundary_score="entropy",
                            boundary_pool_quantile=0.4,
                            boundary_cluster_method="kmeans",
                            min_proto_per_class=4,
                            budget_alpha=0.5,
                            shadow_policy="rank_adaptive",
                            rank_adaptive_global_cap=True,
                            max_total_condensed_ratio=0.12,
                            assignment_chunk_size=8192,
                            inference_dst_chunk_size=250_000,
                            demand_edge_chunk_size=250_000,
                            inference_edge_chunk_size=250_000,
                        )
                    except Exception as exc:
                        status = exception_status(exc)
                        summary = {
                            "dataset": dataset,
                            "variant": variant.name,
                            "seed": args.seed,
                            "ratio": ratio,
                            "status": status,
                            "reason": str(exc),
                            "compiled_head": variant.compiled_head,
                            "label_affinity": variant.label_affinity,
                            "boundary_prototypes": variant.boundary_prototypes,
                            "diffusion_enabled": False,
                            "diffusion_status": "diagnostic_only",
                        }
                        write_json_summary(log_path, summary)
                rows.append(summary_to_lad_row(summary, dataset=dataset, ratio=ratio, variant=variant.name, log_path=log_path))
    write_csv(args.output, rows, LAD_TABLE_FIELDS)


if __name__ == "__main__":
    main()
