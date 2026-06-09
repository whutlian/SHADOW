from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import DATASET_LOSS, lad_feature_mode, write_csv
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


SMALL_DATASETS = ["acm", "dblp", "imdb"]
MEDIUM_DATASETS = ["ogbn-arxiv", "ogbn-products"]
SMALL_FULL_NODE_RATIOS = [0.012, 0.024, 0.048, 0.096]
MEDIUM_FULL_NODE_RATIOS = [0.0005, 0.0025, 0.005]


def _ratio_label(ratio: float) -> str:
    return str(float(ratio)).replace(".", "p")


def _load_dataset(dataset: str, *, download: bool):
    if dataset in SMALL_DATASETS:
        return load_processed_small_dataset(dataset)
    return load_ogb_node_property_dataset(dataset, download=download)


def _allocate_integer_budget(total: int, weights: dict[object, float], *, min_each: int = 1) -> dict[object, int]:
    keys = list(weights)
    if not keys:
        return {}
    total = max(0, int(total))
    if total == 0:
        return {key: 0 for key in keys}
    if total <= len(keys) * min_each:
        out = {key: 0 for key in keys}
        for key in keys[:total]:
            out[key] = 1
        return out
    out = {key: min_each for key in keys}
    remaining = total - sum(out.values())
    weight_sum = sum(max(float(value), 0.0) for value in weights.values())
    if weight_sum <= 0.0:
        weights = {key: 1.0 for key in keys}
        weight_sum = float(len(keys))
    quotas = {key: remaining * max(float(weights[key]), 0.0) / weight_sum for key in keys}
    for key in keys:
        grant = int(math.floor(quotas[key]))
        out[key] += grant
        remaining -= grant
    for key in sorted(keys, key=lambda item: (-(quotas[item] - math.floor(quotas[item])), str(item))):
        if remaining <= 0:
            break
        out[key] += 1
        remaining -= 1
    return out


def _budget_for_full_node_ratio(graph, desired_ratio: float, *, min_proto_per_class: int) -> tuple[int, dict, dict]:
    original_nodes = int(sum(graph.num_nodes.values()))
    total_budget = max(1, int(round(float(desired_ratio) * original_nodes)))
    labels = graph.labels[graph.train_idx]
    num_classes = int(labels[labels >= 0].unique().numel())
    min_target = max(1, num_classes * int(min_proto_per_class))
    target_type = graph.target_type
    relations = list(graph.relations)
    tt = [rel for rel in relations if rel.source_type == target_type and rel.destination_type == target_type]
    nt = [rel for rel in relations if rel not in tt]
    shadow_multiplier = (0.5 if tt else 0.0) + (1.0 if nt else 0.0)
    target_budget = int(round(total_budget / (1.0 + shadow_multiplier))) if shadow_multiplier > 0 else total_budget
    target_budget = max(min_target, min(target_budget, max(min_target, total_budget - len(relations))))
    shadow_total = max(0, total_budget - target_budget)
    weights = {}
    if tt:
        for rel in tt:
            weights[rel] = 0.5 / len(tt)
    if nt:
        for rel in nt:
            weights[rel] = 1.0 / len(nt)
    shadow_budget = _allocate_integer_budget(shadow_total, weights, min_each=1)
    metadata = {
        "requested_full_condensed_node_ratio": float(desired_ratio),
        "planned_total_condensed_nodes": int(total_budget),
        "planned_target_budget": int(target_budget),
        "planned_shadow_total_budget": int(sum(shadow_budget.values())),
        "planned_shadow_budget_by_relation": {str(rel): int(value) for rel, value in shadow_budget.items()},
        "original_nodes_total": original_nodes,
    }
    return target_budget, shadow_budget, metadata


def _run_one(graph, dataset: str, desired_ratio: float, args) -> dict:
    is_medium = dataset in MEDIUM_DATASETS
    min_proto_per_class = 1 if is_medium else 4
    target_budget, shadow_budget, budget_meta = _budget_for_full_node_ratio(
        graph,
        desired_ratio,
        min_proto_per_class=min_proto_per_class,
    )
    log_path = Path(args.log_dir) / f"{dataset}_V2_compiled_plus_lad_fullnode_r{_ratio_label(desired_ratio)}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    try:
        graph_model = "shadow_fusion" if (dataset == "imdb" or is_medium) else "relation_linear"
        use_metapath = dataset in SMALL_DATASETS
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-L",
            stage="lad_full_node_ratio",
            seed=args.seed,
            epochs=args.epochs,
            budget_mode="count",
            target_budget=target_budget,
            feature_dim=128 if is_medium else 64,
            projection_type="random" if is_medium else "raw",
            model_type=graph_model,
            hidden_dim=128,
            dropout=0.3,
            lr=0.03,
            weight_decay=1e-4,
            loss_type=DATASET_LOSS[dataset],
            logit_adjustment_tau=1.0,
            feature_mode=lad_feature_mode(label_affinity=True, metapath=use_metapath),
            metapath_model_input=use_metapath,
            metapath_signature=use_metapath,
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
            compiled_demand_source="shadow_reconstructed",
            boundary_prototypes=False,
            min_proto_per_class=min_proto_per_class,
            budget_alpha=0.5,
            shadow_policy="fixed",
            M_r=shadow_budget,
            assignment_chunk_size=8192 if is_medium else None,
            inference_dst_chunk_size=250_000 if is_medium else None,
            demand_edge_chunk_size=250_000 if is_medium else None,
            inference_edge_chunk_size=250_000 if is_medium else None,
        )
        summary.update(budget_meta)
        summary["actual_full_condensed_node_ratio"] = summary.get("total_condensed_node_ratio")
        summary["actual_full_condensed_node_ratio_error"] = (
            float(summary["actual_full_condensed_node_ratio"]) - float(desired_ratio)
            if summary.get("actual_full_condensed_node_ratio") is not None
            else None
        )
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        status = exception_status(exc)
        payload = {
            "dataset": dataset,
            "variant": "V2_compiled_plus_lad",
            "seed": args.seed,
            "status": status,
            "reason": str(exc),
            "diffusion_enabled": False,
            "diffusion_status": "diagnostic_only",
            **budget_meta,
        }
        write_json_summary(log_path, payload)
        return payload


def _row(summary: dict, dataset: str, desired_ratio: float) -> dict:
    return {
        "dataset": dataset,
        "variant": "V2_compiled_plus_lad",
        "seed": summary.get("seed", 42),
        "requested_full_condensed_node_ratio": desired_ratio,
        "actual_full_condensed_node_ratio": summary.get("total_condensed_node_ratio", summary.get("actual_full_condensed_node_ratio", "")),
        "actual_ratio_error": summary.get("actual_full_condensed_node_ratio_error", ""),
        "planned_total_condensed_nodes": summary.get("planned_total_condensed_nodes", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "effective_M_tau": summary.get("effective_M_tau", ""),
        "shadow_nodes_total": summary.get("shadow_nodes_total", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "byte_size_compression": summary.get("byte_size_compression", ""),
        "status": summary.get("status", "completed"),
        "reason": summary.get("reason", ""),
        "source_log": summary.get("source_log", ""),
    }


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# LAD V2 Full-Graph Condensed Node Ratio Sweep",
        "",
        "Single seed 42. All rows are no-diffusion LAD V2 (`compiled_plus_lad`).",
        "",
        "| Dataset | Requested full node ratio | Actual full node ratio | Acc | Macro-F1 | Condensed nodes | Target prototypes | Shadow nodes | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(value, scale=1.0):
            if value in ("", None):
                return ""
            return f"{float(value) * scale:.4f}"
        lines.append(
            f"| {row['dataset']} | {fmt(row['requested_full_condensed_node_ratio'], 100)}% | "
            f"{fmt(row['actual_full_condensed_node_ratio'], 100)}% | {fmt(row['accuracy'])} | "
            f"{fmt(row['macro_f1'])} | {row.get('condensed_nodes_total','')} | "
            f"{row.get('effective_M_tau','')} | {row.get('shadow_nodes_total','')} | {row.get('status','')} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`", f"- Report: `{path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAD V2 by requested full-graph condensed node ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/lad_full_node_ratio_seed42")
    parser.add_argument("--output", default="experiments/tables/lad_full_node_ratio_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/lad_full_node_ratio_seed42.md")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in SMALL_DATASETS:
        graph = _load_dataset(dataset, download=args.download)
        for ratio in SMALL_FULL_NODE_RATIOS:
            summary = _run_one(graph, dataset, ratio, args)
            rows.append(_row(summary, dataset, ratio))
    for dataset in MEDIUM_DATASETS:
        graph = _load_dataset(dataset, download=args.download)
        for ratio in MEDIUM_FULL_NODE_RATIOS:
            summary = _run_one(graph, dataset, ratio, args)
            rows.append(_row(summary, dataset, ratio))
    output = Path(args.output)
    write_csv(output, rows)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
