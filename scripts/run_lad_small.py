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
    SMALL_RATIOS,
    lad_feature_mode,
    ratio_label,
    summary_to_lad_row,
    write_csv,
)
from shadow_hgc.data.small import load_processed_small_dataset
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
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-L LAD small matrix.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output", default="experiments/tables/lad_stage_small_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/lad_stage_small_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compiled-hidden-dim", type=int, default=256)
    parser.add_argument("--compiled-dropout", type=float, default=0.3)
    parser.add_argument("--compiled-block-gate", type=_bool_arg, default=True)
    args = parser.parse_args()

    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        graph = load_processed_small_dataset(dataset)
        graph_model = "shadow_fusion" if dataset == "imdb" else "relation_linear"
        use_metapath = dataset in {"acm", "dblp", "imdb"}
        relation_gate = dataset == "imdb"
        block_norm = "standardize" if dataset == "imdb" else "none"
        for ratio in SMALL_RATIOS[dataset]:
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
                            feature_dim=64,
                            projection_type="raw",
                            model_type=graph_model,
                            hidden_dim=128,
                            dropout=0.3,
                            lr=0.03,
                            weight_decay=1e-4,
                            loss_type=DATASET_LOSS[dataset],
                            feature_mode=lad_feature_mode(label_affinity=variant.label_affinity, metapath=use_metapath),
                            metapath_model_input=use_metapath,
                            metapath_signature=use_metapath,
                            shadow_policy="rank_adaptive",
                            adaptive_b=dataset == "imdb",
                            relation_gate=relation_gate,
                            block_norm=block_norm,
                            diffusion_enabled=False,
                            diffusion_status="diagnostic_only",
                            label_affinity=variant.label_affinity,
                            label_affinity_mode="all",
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
