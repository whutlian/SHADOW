from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import DATASET_LOSS, LAD_TABLE_FIELDS, ratio_label, summary_to_lad_row, write_csv
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


DIAGNOSTICS = [
    ("imdb", 0.025, "FullDemandTable-MLP", "full_demand", "exact"),
    ("ogbn-arxiv", 0.12, "FullDemandTable-MLP", "full_demand", "exact"),
    ("ogbn-products", 0.12, "FullDemandTable-MLP", "full_demand", "exact"),
    ("acm", 0.096, "PrototypeOracleDemand-MLP", "prototypes", "prototype_oracle"),
    ("imdb", 0.025, "PrototypeOracleDemand-MLP", "prototypes", "prototype_oracle"),
    ("ogbn-arxiv", 0.12, "PrototypeOracleDemand-MLP", "prototypes", "prototype_oracle"),
    ("ogbn-products", 0.12, "PrototypeOracleDemand-MLP", "prototypes", "prototype_oracle"),
]


def _load_dataset(dataset: str, *, download: bool):
    if dataset in {"acm", "dblp", "imdb"}:
        return load_processed_small_dataset(dataset)
    return load_ogb_node_property_dataset(dataset, download=download)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-L LAD diagnostic upper bounds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="experiments/tables/lad_stage_diagnostics_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/lad_stage_diagnostics_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ratio, method, train_scope, demand_source in DIAGNOSTICS:
        variant = method
        log_path = log_dir / f"{dataset}_{method}_r{ratio_label(ratio)}_seed{args.seed}.json"
        if args.skip_existing and log_path.exists():
            summary = json.loads(log_path.read_text(encoding="utf-8"))
            rows.append(summary_to_lad_row(summary, dataset=dataset, ratio=ratio, variant=variant, log_path=log_path))
            continue
        try:
            graph = _load_dataset(dataset, download=args.download)
            is_medium = dataset.startswith("ogbn-")
            summary = run_shadow_hgc_experiment(
                graph,
                output_path=log_path,
                method_name=method,
                stage="lad_diagnostic",
                seed=args.seed,
                epochs=args.epochs,
                budget_mode="ratio",
                ratio=ratio,
                ratio_base="train_target",
                feature_dim=128 if is_medium else 64,
                projection_type="random" if is_medium else "raw",
                model_type="relation_mlp",
                hidden_dim=128,
                dropout=0.3,
                lr=0.03,
                weight_decay=1e-4,
                loss_type=DATASET_LOSS[dataset],
                logit_adjustment_tau=1.0,
                feature_mode="label_affinity",
                diffusion_enabled=False,
                diffusion_status="diagnostic_only",
                label_affinity=True,
                label_affinity_mode="target_target" if is_medium else "all",
                label_affinity_self_exclude=True,
                label_affinity_block_norm="row_l1",
                compiled_head=True,
                compiled_head_fusion="concat_mlp",
                compiled_hidden_dim=256,
                compiled_dropout=0.3,
                compiled_block_gate=True,
                compiled_train_scope=train_scope,
                compiled_demand_source=demand_source,
                boundary_prototypes=False,
                min_proto_per_class=4,
                budget_alpha=0.5,
                shadow_policy="rank_adaptive" if is_medium else "fixed",
                rank_adaptive_global_cap=is_medium,
                max_total_condensed_ratio=0.12 if is_medium else None,
                assignment_chunk_size=8192 if is_medium else None,
                inference_dst_chunk_size=250_000 if is_medium else None,
                demand_edge_chunk_size=250_000 if is_medium else None,
                inference_edge_chunk_size=250_000 if is_medium else None,
            )
            summary["diagnostic_method"] = method
            write_json_summary(log_path, summary)
        except Exception as exc:
            status = exception_status(exc)
            summary = {
                "dataset": dataset,
                "variant": variant,
                "seed": args.seed,
                "ratio": ratio,
                "status": status,
                "reason": str(exc),
                "compiled_head": True,
                "label_affinity": True,
                "boundary_prototypes": False,
                "diffusion_enabled": False,
                "diffusion_status": "diagnostic_only",
                "diagnostic_method": method,
            }
            write_json_summary(log_path, summary)
        rows.append(summary_to_lad_row(summary, dataset=dataset, ratio=ratio, variant=variant, log_path=log_path))
    write_csv(args.output, rows, LAD_TABLE_FIELDS)


if __name__ == "__main__":
    main()
