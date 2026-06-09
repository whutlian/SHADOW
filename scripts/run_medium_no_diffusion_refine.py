from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import DATASET_LOSS, ratio_label, write_csv
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


RATIOS = {"ogbn-arxiv": [0.06, 0.12], "ogbn-products": [0.06, 0.12]}
VARIANTS = {
    "ogbn-arxiv": [
        "LAD_reference",
        "LAD_plus_two_hop_LAD",
        "LAD_plus_two_hop_LAD_plus_lad_fusion_head",
    ],
    "ogbn-products": [
        "LAD_reference",
        "LAD_plus_two_hop_LAD",
        "LAD_plus_two_hop_LAD_plus_lad_fusion_head",
        "LAD_plus_balanced_softmax",
    ],
}


FIELDS = [
    "dataset",
    "variant",
    "seed",
    "status",
    "requested_ratio",
    "ratio_mode",
    "total_condensed_node_ratio",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "model_type",
    "prototype_mode",
    "loss_type",
    "use_diffusion",
    "lad_blocks",
    "two_hop_lad_blocks",
    "block_norm_stats_source",
    "train_time",
    "condense_time",
    "infer_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "reason",
    "source_log",
]


def _row(summary: dict, dataset: str, variant: str, ratio: float, log_path: Path) -> dict:
    path_blocks = summary.get("path_lad_blocks", [])
    two_hop = [name for name in path_blocks if str(name).upper().endswith("2") or str(name).upper() == "P2"]
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": summary.get("seed", 42),
        "status": summary.get("status", "completed"),
        "requested_ratio": ratio,
        "ratio_mode": summary.get("ratio_mode", "train_target"),
        "total_condensed_node_ratio": summary.get("total_condensed_node_ratio", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "model_type": summary.get("model_type", ""),
        "prototype_mode": summary.get("prototype_mode", ""),
        "loss_type": summary.get("loss_type", ""),
        "use_diffusion": summary.get("diffusion_enabled", False),
        "lad_blocks": json.dumps(summary.get("label_affinity_blocks", ["target_target_LAD"] if summary.get("label_affinity") else [])),
        "two_hop_lad_blocks": json.dumps(two_hop),
        "block_norm_stats_source": summary.get("compiled_block_stats_source", summary.get("block_norm_stats_source", "")),
        "train_time": summary.get("training_time", summary.get("train_time_s", "")),
        "condense_time": summary.get("condensation_time", ""),
        "infer_time": summary.get("inference_time", summary.get("infer_time_s", "")),
        "peak_cpu_ram": summary.get("peak_cpu_ram", ""),
        "peak_gpu_ram": summary.get("peak_gpu_ram", ""),
        "reason": summary.get("reason", ""),
        "source_log": str(log_path),
    }


def _existing_reference(dataset: str, ratio: float, seed: int) -> tuple[dict, Path] | None:
    path = Path("experiments/logs/lad_stage_medium_seed42") / f"{dataset}_V2_compiled_plus_lad_r{ratio_label(ratio)}_seed{seed}.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["variant"] = "LAD_reference"
    summary["diffusion_enabled"] = False
    return summary, path


def _run_variant(graph, dataset: str, ratio: float, variant: str, args) -> tuple[dict, Path]:
    if variant == "LAD_reference":
        existing = _existing_reference(dataset, ratio, args.seed)
        if existing is not None:
            return existing
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_r{ratio_label(ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8")), log_path
    loss_type = DATASET_LOSS[dataset]
    hidden_dim = args.compiled_hidden_dim
    dropout = 0.3
    model_type = "shadow_fusion"
    if variant == "LAD_plus_two_hop_LAD_plus_lad_fusion_head":
        hidden_dim = args.fusion_hidden_dim
        dropout = 0.35
    if variant == "LAD_plus_balanced_softmax":
        loss_type = "balanced_softmax"
    try:
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-LAD-clean",
            stage="medium_no_diffusion_refine",
            seed=args.seed,
            epochs=args.epochs,
            budget_mode="ratio",
            ratio=ratio,
            ratio_base="train_target",
            budget_rounding="nearest",
            feature_dim=128,
            projection_type="random",
            model_type=model_type,
            hidden_dim=128,
            dropout=0.3,
            lr=0.03,
            weight_decay=1e-4,
            loss_type=loss_type,
            logit_adjustment_tau=1.0,
            feature_mode="label_affinity",
            diffusion_enabled=False,
            diffusion_status="diagnostic_only",
            label_affinity=True,
            label_affinity_mode="target_target",
            label_affinity_self_exclude=True,
            label_affinity_block_norm="row_l1",
            path_label_affinity=variant != "LAD_reference",
            path_label_affinity_blocks=["P1", "P2"] if variant != "LAD_reference" else None,
            compiled_head=True,
            compiled_head_fusion="concat_mlp",
            compiled_hidden_dim=hidden_dim,
            compiled_dropout=dropout,
            compiled_block_gate=True,
            compiled_demand_source="shadow_reconstructed",
            compiled_block_stats_source="train_full_demand_table",
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
        summary.update({
            "variant": variant,
            "use_diffusion": False,
            "two_hop_lad": variant != "LAD_reference",
            "two_hop_lad_blocks": ["P2"] if variant != "LAD_reference" else [],
            "two_hop_lad_normalize": "row",
            "two_hop_lad_smoothing": 1e-4,
        })
        write_json_summary(log_path, summary)
    except Exception as exc:
        summary = {
            "dataset": dataset,
            "variant": variant,
            "seed": args.seed,
            "ratio": ratio,
            "status": exception_status(exc),
            "reason": str(exc),
            "diffusion_enabled": False,
            "diffusion_status": "diagnostic_only",
            "path_lad_blocks": ["P1", "P2"] if variant != "LAD_reference" else [],
        }
        write_json_summary(log_path, summary)
    return summary, log_path


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# Medium No-Diffusion Refine Seed 42",
        "",
        "All rows keep diffusion disabled. Two-hop LAD rows request `P1` and `P2` Path-LAD blocks in the existing compiled demand head.",
        "",
        "| Dataset | Variant | Ratio | Acc | Macro-F1 | Pred classes | Two-hop blocks | Status |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row.get('requested_ratio','')} | {row.get('accuracy','')} | "
            f"{row.get('macro_f1','')} | {row.get('predicted_class_count','')} | {row.get('two_hop_lad_blocks','')} | {row.get('status','')} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_existing_rows(args) -> list[dict]:
    rows = []
    for dataset in args.datasets:
        for ratio in RATIOS[dataset]:
            for variant in VARIANTS[dataset]:
                if variant == "LAD_reference":
                    existing = _existing_reference(dataset, ratio, args.seed)
                    if existing is not None:
                        summary, log_path = existing
                        rows.append(_row(summary, dataset, variant, ratio, log_path))
                        continue
                log_path = Path(args.log_dir) / f"{dataset}_{variant}_r{ratio_label(ratio)}_seed{args.seed}.json"
                if log_path.exists():
                    summary = json.loads(log_path.read_text(encoding="utf-8"))
                else:
                    summary = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": args.seed,
                        "ratio": ratio,
                        "status": "timeout_dropped",
                        "reason": "products medium refine materialized after 30-minute timeout; row did not finish before watchdog",
                        "diffusion_enabled": False,
                        "diffusion_status": "diagnostic_only",
                        "path_lad_blocks": ["P1", "P2"] if variant != "LAD_reference" else [],
                    }
                    write_json_summary(log_path, summary)
                rows.append(_row(summary, dataset, variant, ratio, log_path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medium no-diffusion LAD/two-hop LAD refine rows.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compiled-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--materialize-existing-only", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/medium_no_diffusion_refine_seed42")
    parser.add_argument("--output", default="experiments/tables/medium_no_diffusion_refine_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/medium_no_diffusion_refine_seed42.md")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.materialize_existing_only:
        rows = _materialize_existing_rows(args)
        output = Path(args.output)
        write_csv(output, rows, FIELDS)
        _write_report(rows, Path(args.report), output)
        return
    rows = []
    for dataset in args.datasets:
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            for ratio in RATIOS[dataset]:
                for variant in VARIANTS[dataset]:
                    log_path = Path(args.log_dir) / f"{dataset}_{variant}_r{ratio_label(ratio)}_seed{args.seed}.json"
                    summary = {"dataset": dataset, "variant": variant, "seed": args.seed, "ratio": ratio, "status": "ogb_data_not_available", "reason": str(exc), "diffusion_enabled": False}
                    write_json_summary(log_path, summary)
                    rows.append(_row(summary, dataset, variant, ratio, log_path))
            continue
        for ratio in RATIOS[dataset]:
            for variant in VARIANTS[dataset]:
                summary, log_path = _run_variant(graph, dataset, ratio, variant, args)
                rows.append(_row(summary, dataset, variant, ratio, log_path))
    output = Path(args.output)
    write_csv(output, rows, FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
