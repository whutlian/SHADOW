from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_full_node_ratio import _budget_for_full_node_ratio
from scripts.run_sota_common import (
    MEDIUM_SOTA_FULL_NODE_RATIOS,
    PATH_LAD_BLOCKS,
    SOTA_MEDIUM_VARIANTS,
    SOTA_TABLE_FIELDS,
    SOTAVariant,
    read_summary,
    sota_loss,
    sota_ratio_label,
    summary_to_sota_row,
    write_csv,
)
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _run_one(graph, dataset: str, full_ratio: float, variant: SOTAVariant, args) -> tuple[dict, Path]:
    target_budget, shadow_budget, budget_meta = _budget_for_full_node_ratio(
        graph,
        full_ratio,
        min_proto_per_class=1,
    )
    log_path = Path(args.log_dir) / f"{dataset}_{variant.name}_fullnode_r{sota_ratio_label(full_ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return read_summary(log_path), log_path
    try:
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-SOTA",
            stage="sota",
            seed=args.seed,
            epochs=args.epochs,
            budget_mode="count",
            target_budget=target_budget,
            M_r=shadow_budget,
            feature_dim=128,
            projection_type="random",
            model_type="shadow_fusion",
            hidden_dim=128,
            dropout=0.3,
            lr=0.03,
            weight_decay=1e-4,
            loss_type=sota_loss(dataset, variant),
            logit_adjustment_tau=1.0,
            feature_mode="label_affinity" if variant.label_affinity else "base",
            diffusion_enabled=False,
            diffusion_status="diagnostic_only",
            label_affinity=variant.label_affinity,
            label_affinity_mode="target_target",
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
            min_proto_per_class=1,
            budget_alpha=0.5,
            shadow_policy="fixed",
            rank_adaptive_global_cap=True,
            max_total_condensed_ratio=max(0.12, full_ratio),
            assignment_chunk_size=8192,
            inference_dst_chunk_size=250_000,
            demand_edge_chunk_size=250_000,
            inference_edge_chunk_size=250_000,
        )
        summary.update(budget_meta)
        summary["variant"] = variant.name
        summary["requested_full_condensed_node_ratio"] = full_ratio
        summary["actual_full_condensed_node_ratio"] = summary.get("total_condensed_node_ratio", "")
        summary["sota_backbone"] = "sign_lad_mlp" if variant.use_kd else "compiled_demand_head"
        write_json_summary(log_path, summary)
    except Exception as exc:
        status = exception_status(exc)
        summary = {
            "dataset": dataset,
            "variant": variant.name,
            "seed": args.seed,
            "requested_full_condensed_node_ratio": full_ratio,
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
            **budget_meta,
        }
        write_json_summary(log_path, summary)
    return summary, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-SOTA medium full-node-ratio matrix.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="experiments/tables/sota_medium_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/sota_medium_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mark-missing-timeout", action="store_true")
    args = parser.parse_args()

    rows = []
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            for full_ratio in MEDIUM_SOTA_FULL_NODE_RATIOS[dataset]:
                for variant in SOTA_MEDIUM_VARIANTS:
                    log_path = Path(args.log_dir) / f"{dataset}_{variant.name}_fullnode_r{sota_ratio_label(full_ratio)}_seed{args.seed}.json"
                    summary = {
                        "dataset": dataset,
                        "variant": variant.name,
                        "seed": args.seed,
                        "requested_full_condensed_node_ratio": full_ratio,
                        "status": "ogb_data_not_available",
                        "reason": str(exc),
                    }
                    write_json_summary(log_path, summary)
                    rows.append(summary_to_sota_row(summary, dataset=dataset, variant=variant, log_path=log_path, requested_full_ratio=full_ratio))
            continue
        for full_ratio in MEDIUM_SOTA_FULL_NODE_RATIOS[dataset]:
            for variant in SOTA_MEDIUM_VARIANTS:
                if args.mark_missing_timeout:
                    _, _, budget_meta = _budget_for_full_node_ratio(graph, full_ratio, min_proto_per_class=1)
                    log_path = Path(args.log_dir) / f"{dataset}_{variant.name}_fullnode_r{sota_ratio_label(full_ratio)}_seed{args.seed}.json"
                    if log_path.exists():
                        summary = read_summary(log_path)
                    else:
                        summary = {
                            "dataset": dataset,
                            "variant": variant.name,
                            "seed": args.seed,
                            "requested_full_condensed_node_ratio": full_ratio,
                            "status": "timeout_dropped",
                            "reason": "products SOTA medium row was dropped after no new JSON progress during the timeout window",
                            "method_family": "sota",
                            "prototype_mode": variant.prototype_mode,
                            "source_anchor_mode": variant.source_anchor_mode,
                            "teacher_type": variant.teacher_type,
                            "use_kd": variant.use_kd,
                            "path_lad_blocks": [],
                            "diffusion_enabled": False,
                            "diffusion_status": "diagnostic_only",
                            **budget_meta,
                        }
                        write_json_summary(log_path, summary)
                else:
                    summary, log_path = _run_one(graph, dataset, full_ratio, variant, args)
                rows.append(summary_to_sota_row(summary, dataset=dataset, variant=variant, log_path=log_path, requested_full_ratio=full_ratio))
    write_csv(args.output, rows, SOTA_TABLE_FIELDS)


if __name__ == "__main__":
    main()
