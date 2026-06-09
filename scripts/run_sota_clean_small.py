from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import DATASET_LOSS
from scripts.run_rpp_common import base_row
from scripts.run_sota_common import SOTAVariant, sota_ratio_label, summary_to_sota_row, write_csv
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment
from shadow_hgc.train.sehgnn_lite_target import build_schema_default_blocks, train_prototype_sehgnn_lite


CLEAN_FIELDS = [
    "dataset",
    "variant",
    "seed",
    "status",
    "invalid_reasons",
    "requested_ratio",
    "ratio_mode",
    "total_condensed_node_ratio",
    "total_condensed_edge_ratio",
    "byte_size_compression",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "model_type",
    "prototype_mode",
    "loss_type",
    "teacher_type",
    "use_kd",
    "use_diffusion",
    "use_source_anchors",
    "use_coverage_medoid",
    "metapath_blocks",
    "path_lad_blocks",
    "lad_blocks",
    "two_hop_lad_blocks",
    "block_norm_stats_source",
    "train_time",
    "condense_time",
    "infer_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "source_log",
]


def _common_clean_summary(summary: dict, *, dataset: str, variant: str, seed: int, log_path: Path, requested_ratio: float) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "status": summary.get("status", "completed"),
        "invalid_reasons": summary.get("invalid_reasons", []),
        "requested_ratio": requested_ratio,
        "ratio_mode": summary.get("ratio_mode", "train_target"),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "total_condensed_edge_ratio": summary.get("total_condensed_edge_ratio", ""),
        "byte_size_compression": summary.get("byte_size_compression", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", ""),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "model_type": summary.get("model_type", ""),
        "prototype_mode": summary.get("prototype_mode", ""),
        "loss_type": summary.get("loss_type", ""),
        "teacher_type": summary.get("teacher_type", "none"),
        "use_kd": summary.get("use_kd", False),
        "use_diffusion": summary.get("use_diffusion", summary.get("diffusion_enabled", False)),
        "use_source_anchors": summary.get("use_source_anchors", summary.get("source_anchor_mode", "none") != "none"),
        "use_coverage_medoid": summary.get("prototype_mode", "") == "coverage_medoid",
        "metapath_blocks": json.dumps(summary.get("metapath_blocks", [])),
        "path_lad_blocks": json.dumps(summary.get("path_lad_blocks", [])),
        "lad_blocks": json.dumps(summary.get("lad_blocks", [])),
        "two_hop_lad_blocks": json.dumps(summary.get("two_hop_lad_blocks", [])),
        "block_norm_stats_source": summary.get("block_norm_stats_source", summary.get("compiled_block_stats_source", "")),
        "train_time": summary.get("train_time", summary.get("training_time", "")),
        "condense_time": summary.get("condense_time", summary.get("condensation_time", "")),
        "infer_time": summary.get("infer_time", summary.get("inference_time", "")),
        "peak_cpu_ram": summary.get("peak_cpu_ram", ""),
        "peak_gpu_ram": summary.get("peak_gpu_ram", ""),
        "source_log": str(log_path),
    }


def _run_pipeline_reference(graph, dataset: str, ratio: float, variant: str, args) -> tuple[dict, Path]:
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_r{sota_ratio_label(ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8")), log_path
    try:
        if dataset == "imdb":
            model_type = "shadow_fusion"
            feature_mode = "metapath"
            block_norm = "standardize"
            relation_gate = True
            adaptive_b = True
            shadow_policy = "rank_adaptive"
        else:
            model_type = "relation_linear"
            feature_mode = "base"
            block_norm = "none"
            relation_gate = False
            adaptive_b = False
            shadow_policy = "fixed"
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-R1-reference" if dataset != "imdb" else "Shadow-HGC-R++-reference",
            stage="sota_alignment_clean",
            seed=args.seed,
            epochs=args.epochs,
            budget_mode="ratio",
            ratio=ratio,
            ratio_base="train_target",
            feature_dim=64,
            projection_type="raw",
            model_type=model_type,
            hidden_dim=128,
            dropout=0.3,
            lr=0.03,
            weight_decay=1e-4,
            loss_type=DATASET_LOSS[dataset],
            feature_mode=feature_mode,
            metapath_model_input=dataset == "imdb",
            metapath_signature=dataset == "imdb",
            diffusion_enabled=False,
            diffusion_status="diagnostic_only",
            shadow_policy=shadow_policy,
            adaptive_b=adaptive_b,
            relation_gate=relation_gate,
            block_norm=block_norm,
            min_proto_per_class=4,
            budget_alpha=0.5,
        )
        summary.update({"variant": variant, "teacher_type": "none", "use_kd": False, "use_diffusion": False})
        write_json_summary(log_path, summary)
    except Exception as exc:
        summary = {"dataset": dataset, "variant": variant, "seed": args.seed, "status": exception_status(exc), "reason": str(exc), "use_diffusion": False}
        write_json_summary(log_path, summary)
    return summary, log_path


def _run_clean_sehgnn(graph, dataset: str, ratio: float, variant: str, args, *, include_path_lad_v2: bool = False, include_self: bool = True) -> tuple[dict, Path]:
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_r{sota_ratio_label(ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8")), log_path
    try:
        blocks, metadata = build_schema_default_blocks(
            graph,
            include_self=include_self,
            include_metapath=not include_path_lad_v2,
            include_path_lad_v2=include_path_lad_v2,
        )
        if not blocks:
            raise RuntimeError("no feature blocks available for clean SeHGNNLite row")
        run = train_prototype_sehgnn_lite(
            graph,
            blocks=blocks,
            metadata=metadata,
            requested_ratio=ratio,
            seed=args.seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            dropout=0.3,
            lr=0.01,
            weight_decay=1e-4,
            loss_type=DATASET_LOSS[dataset],
            min_proto_per_class=4,
        )
        summary = {
            "dataset": dataset,
            "variant": variant,
            "seed": args.seed,
            "status": "completed",
            "target_type": graph.target_type,
            "teacher_type": "none",
            "use_kd": False,
            "use_diffusion": False,
            "use_source_anchors": False,
            "use_coverage_medoid": False,
            **run.summary,
        }
        write_json_summary(log_path, summary)
    except Exception as exc:
        summary = {"dataset": dataset, "variant": variant, "seed": args.seed, "status": exception_status(exc), "reason": str(exc), "use_diffusion": False}
        write_json_summary(log_path, summary)
    return summary, log_path


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# SOTA Clean Small Seed 42",
        "",
        "Clean rows disable KD, coverage medoids, source anchors, and diffusion. S1 rows use actual `SeHGNNLite` feature-block training.",
        "",
        "| Dataset | Variant | Ratio | Acc | Macro-F1 | Pred classes | Condensed node ratio | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row.get('requested_ratio','')} | {row.get('accuracy','')} | "
            f"{row.get('macro_f1','')} | {row.get('predicted_class_count','')} | {row.get('total_condensed_node_ratio','')} | {row.get('status','')} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- DBLP clean SeHGNN is author-targeted and uses APA when available.",
        "- IMDB Path-LAD v2 rows use train labels only with leave-one-out, row normalization, hub clipping diagnostics, and no exposed meta-path edge types.",
        "- `PathLAD_v2_plus_shadow_fusion` is not promoted unless later wired into the graph shadow-fusion model; this script records it as a feature-block fusion diagnostic.",
        "",
        f"- CSV: `{csv_path}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean SOTA small candidates, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/sota_clean_small_seed42")
    parser.add_argument("--output", default="experiments/tables/sota_clean_small_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/sota_clean_small_seed42.md")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset, ratios in {"acm": [0.048, 0.096, 0.12, 0.15]}.items():
        graph = load_processed_small_dataset(dataset)
        for ratio in ratios:
            summary, log_path = _run_clean_sehgnn(graph, dataset, ratio, "S1_clean_metapath_sehgnn", args)
            rows.append(_common_clean_summary(summary, dataset=dataset, variant="S1_clean_metapath_sehgnn", seed=args.seed, log_path=log_path, requested_ratio=ratio))

    graph = load_processed_small_dataset("dblp")
    for ratio in [0.005, 0.065, 0.096]:
        summary, log_path = _run_pipeline_reference(graph, "dblp", ratio, "S0_current_best", args)
        rows.append(_common_clean_summary(summary, dataset="dblp", variant="S0_current_best", seed=args.seed, log_path=log_path, requested_ratio=ratio))
        summary, log_path = _run_clean_sehgnn(graph, "dblp", ratio, "S1_clean_APA_sehgnn", args)
        rows.append(_common_clean_summary(summary, dataset="dblp", variant="S1_clean_APA_sehgnn", seed=args.seed, log_path=log_path, requested_ratio=ratio))

    graph = load_processed_small_dataset("imdb")
    for ratio in [0.005, 0.025, 0.05]:
        summary, log_path = _run_pipeline_reference(graph, "imdb", ratio, "Rpp_shadow_fusion_class_balanced_reference", args)
        rows.append(_common_clean_summary(summary, dataset="imdb", variant="Rpp_shadow_fusion_class_balanced_reference", seed=args.seed, log_path=log_path, requested_ratio=ratio))
        summary, log_path = _run_clean_sehgnn(graph, "imdb", ratio, "S1_clean_MAM_MDM_MKM", args)
        rows.append(_common_clean_summary(summary, dataset="imdb", variant="S1_clean_MAM_MDM_MKM", seed=args.seed, log_path=log_path, requested_ratio=ratio))
        summary, log_path = _run_clean_sehgnn(graph, "imdb", ratio, "PathLAD_v2_only", args, include_path_lad_v2=True, include_self=False)
        rows.append(_common_clean_summary(summary, dataset="imdb", variant="PathLAD_v2_only", seed=args.seed, log_path=log_path, requested_ratio=ratio))
        summary, log_path = _run_clean_sehgnn(graph, "imdb", ratio, "PathLAD_v2_plus_shadow_fusion", args, include_path_lad_v2=True, include_self=True)
        rows.append(_common_clean_summary(summary, dataset="imdb", variant="PathLAD_v2_plus_shadow_fusion", seed=args.seed, log_path=log_path, requested_ratio=ratio))

    output = Path(args.output)
    write_csv(output, rows, CLEAN_FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
