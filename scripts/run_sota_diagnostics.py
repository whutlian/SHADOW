from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sota_common import (
    PATH_LAD_BLOCKS,
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


DIAGNOSTICS: tuple[tuple[str, float, str, SOTAVariant, dict], ...] = (
    ("acm", 0.096, "Medoid-vs-Mean", SOTA_SMALL_VARIANTS[1], {"prototype_mode": "kmeans_mean"}),
    ("acm", 0.096, "CoverageMedoid", SOTA_SMALL_VARIANTS[2], {}),
    ("imdb", 0.048, "PathLAD-off", replace(SOTA_SMALL_VARIANTS[3], path_label_affinity=False), {}),
    ("imdb", 0.048, "PathLAD-on", SOTA_SMALL_VARIANTS[3], {}),
    ("imdb", 0.048, "KD-off", SOTA_SMALL_VARIANTS[3], {}),
    ("imdb", 0.048, "KD-on", SOTA_SMALL_VARIANTS[4], {}),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused Shadow-HGC-SOTA diagnostics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="experiments/tables/sota_diagnostics_seed42.csv")
    parser.add_argument("--log-dir", default="experiments/logs/sota_diagnostics_seed42")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    graphs = {}
    for dataset, ratio, diagnostic_name, variant, overrides in DIAGNOSTICS:
        graph = graphs.setdefault(dataset, load_processed_small_dataset(dataset))
        log_path = Path(args.log_dir) / f"{dataset}_{diagnostic_name}_r{sota_ratio_label(ratio)}_seed{args.seed}.json"
        if args.skip_existing and log_path.exists():
            summary = read_summary(log_path)
        else:
            try:
                use_metapath = dataset in {"acm", "dblp", "imdb"}
                graph_model = "shadow_fusion" if dataset == "imdb" else "relation_linear"
                prototype_mode = overrides.get("prototype_mode", variant.prototype_mode)
                summary = run_shadow_hgc_experiment(
                    graph,
                    output_path=log_path,
                    method_name="Shadow-HGC-SOTA-Diagnostic",
                    stage="sota",
                    seed=args.seed,
                    epochs=args.epochs,
                    budget_mode="ratio",
                    ratio=ratio,
                    ratio_base="train_target",
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
                    prototype_mode=prototype_mode,
                    source_anchor_mode=variant.source_anchor_mode,
                    teacher_type=variant.teacher_type,
                    use_kd=variant.use_kd,
                    kd_weight=0.5,
                    min_proto_per_class=4,
                    budget_alpha=0.5,
                )
                summary["variant"] = diagnostic_name
                summary["diagnostic_name"] = diagnostic_name
                write_json_summary(log_path, summary)
            except Exception as exc:
                summary = {
                    "dataset": dataset,
                    "variant": diagnostic_name,
                    "seed": args.seed,
                    "ratio": ratio,
                    "status": exception_status(exc),
                    "reason": str(exc),
                    "prototype_mode": variant.prototype_mode,
                    "source_anchor_mode": variant.source_anchor_mode,
                    "teacher_type": variant.teacher_type,
                    "use_kd": variant.use_kd,
                }
                write_json_summary(log_path, summary)
        rows.append(summary_to_sota_row(summary, dataset=dataset, variant=diagnostic_name, log_path=log_path, requested_ratio=ratio))
    write_csv(args.output, rows, SOTA_TABLE_FIELDS)


if __name__ == "__main__":
    main()
