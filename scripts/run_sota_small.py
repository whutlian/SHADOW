from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sota_common import (
    PATH_LAD_BLOCKS,
    SMALL_SOTA_RATIOS,
    SOTA_SMALL_VARIANTS,
    SOTA_TABLE_FIELDS,
    SOTAVariant,
    read_summary,
    sota_loss,
    sota_ratio_label,
    summary_to_sota_row,
    write_csv,
)
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _run_one(graph, dataset: str, ratio: float, variant: SOTAVariant, args) -> tuple[dict, Path]:
    log_path = Path(args.log_dir) / f"{dataset}_{variant.name}_r{sota_ratio_label(ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return read_summary(log_path), log_path
    use_metapath = dataset in {"acm", "dblp", "imdb"}
    graph_model = "shadow_fusion" if dataset == "imdb" else "relation_linear"
    try:
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-SOTA",
            stage="sota",
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
            loss_type=sota_loss(dataset, variant),
            feature_mode="metapath" if use_metapath else "base",
            metapath_model_input=use_metapath,
            metapath_signature=use_metapath,
            shadow_policy="rank_adaptive",
            adaptive_b=dataset == "imdb",
            relation_gate=dataset == "imdb",
            block_norm="standardize" if dataset == "imdb" else "none",
            diffusion_enabled=False,
            diffusion_status="diagnostic_only",
            label_affinity=variant.label_affinity,
            label_affinity_mode="all",
            label_affinity_self_exclude=True,
            label_affinity_block_norm="row_l1",
            path_label_affinity=variant.path_label_affinity,
            path_label_affinity_blocks=PATH_LAD_BLOCKS[dataset],
            compiled_head=variant.compiled_head,
            compiled_head_fusion="concat_mlp",
            compiled_hidden_dim=256,
            compiled_dropout=0.3,
            compiled_block_gate=True,
            compiled_demand_source="shadow_reconstructed",
            compiled_block_stats_source="train_full_demand_table",
            prototype_mode=variant.prototype_mode,
            source_anchor_mode=variant.source_anchor_mode,
            teacher_type=variant.teacher_type,
            use_kd=variant.use_kd,
            kd_temperature=2.0,
            kd_weight=0.5,
            boundary_prototypes=variant.boundary_prototypes,
            boundary_fraction=0.3,
            boundary_score="entropy",
            boundary_pool_quantile=0.4,
            boundary_cluster_method="kmeans",
            min_proto_per_class=4,
            budget_alpha=0.5,
        )
        summary["variant"] = variant.name
        summary["sota_backbone"] = "sehgnn_lite" if variant.compiled_head else "current_best"
        summary["sota_components"] = {
            "metapath_blocks": bool(use_metapath and variant.name != "S0_current_best"),
            "coverage_medoids": variant.prototype_mode != "kmeans_mean",
            "path_lad": variant.path_label_affinity,
            "source_anchors": variant.source_anchor_mode != "none",
            "teacher_kd": variant.use_kd,
        }
        write_json_summary(log_path, summary)
    except Exception as exc:
        status = exception_status(exc)
        summary = {
            "dataset": dataset,
            "variant": variant.name,
            "seed": args.seed,
            "ratio": ratio,
            "status": status,
            "reason": str(exc),
            "method_family": "sota",
            "prototype_mode": variant.prototype_mode,
            "source_anchor_mode": variant.source_anchor_mode,
            "teacher_type": variant.teacher_type,
            "use_kd": variant.use_kd,
            "path_lad_blocks": [],
            "diffusion_enabled": False,
            "diffusion_status": "diagnostic_only",
        }
        write_json_summary(log_path, summary)
    return summary, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-SOTA small seed-42 matrix.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="experiments/tables/sota_small_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/sota_small_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        graph = load_processed_small_dataset(dataset)
        for ratio in SMALL_SOTA_RATIOS[dataset]:
            for variant in SOTA_SMALL_VARIANTS:
                summary, log_path = _run_one(graph, dataset, ratio, variant, args)
                rows.append(summary_to_sota_row(summary, dataset=dataset, variant=variant, log_path=log_path, requested_ratio=ratio))
    write_csv(args.output, rows, SOTA_TABLE_FIELDS)


if __name__ == "__main__":
    main()
