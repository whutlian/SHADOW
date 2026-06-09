from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset_full_schema
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.logit_propagation import propagate_logits
from shadow_hgc.features.metapath_table import compute_metapath_feature
from shadow_hgc.features.scap_v2 import compute_target_target_scap_v2
from shadow_hgc.features.structure_stats import compute_structure_stats_block
from shadow_hgc.features.typed_feature_demand import compute_typed_feature_demand
from shadow_hgc.fullgraph.metapath_specs import available_metapath_specs
from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv, write_json
from shadow_hgc.fullgraph.sfb_v2_train import format_allocation_failure, should_run_medium_row, train_sfb_v2_table_model


GATES = {"acm": 0.93, "dblp": 0.91, "imdb": 0.60, "ogbn-arxiv": 0.70, "ogbn-products": 0.72}
RECOVERY_GATES = {"dblp": 0.84, "imdb": 0.45}
VARIANTS = ["B0_self", "B1_typed_demand", "B2_metapath", "B3_scap_v2", "B4_logit_prop"]
FIELDS = [
    "dataset",
    "variant",
    "seed",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "prediction_entropy",
    "gate_acc",
    "gate_acc_passed",
    "recovery_gate",
    "recovery_gate_passed",
    "model_type",
    "enabled_blocks",
    "block_dims",
    "block_gates",
    "metapath_blocks",
    "skipped_metapaths",
    "uses_diffusion",
    "uses_dense_p2",
    "uses_dense_metapath_adjacency",
    "uses_full_graph_backprop",
    "uses_e_by_d_materialization",
    "cache_all_targets",
    "edge_scans_by_block",
    "cache_bytes_by_block",
    "disk_bytes",
    "training_time_s",
    "inference_time_s",
    "wall_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "medium_execution_mode",
    "allocation_failure",
    "source_log",
]


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return int(valid.max().item()) + 1 if valid.numel() else 0


def _load_graph(dataset: str):
    if dataset in {"acm", "dblp", "imdb"}:
        return load_processed_small_dataset_full_schema(dataset)
    return load_ogb_node_property_dataset(dataset, root="dataset", download=False)


def _target_rows(graph) -> torch.Tensor:
    return torch.arange(graph.num_nodes[graph.target_type], dtype=torch.long)


def _slice_edges(edge_index: torch.Tensor, edge_limit: int) -> torch.Tensor:
    if int(edge_limit) <= 0 or int(edge_index.shape[1]) <= int(edge_limit):
        return edge_index
    return edge_index[:, : int(edge_limit)].contiguous()


def _train_rows_for_stats(graph, max_train_rows: int) -> torch.Tensor:
    if int(max_train_rows) <= 0 or graph.train_idx.numel() <= int(max_train_rows):
        return graph.train_idx
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(graph.train_idx.numel(), generator=generator)[: int(max_train_rows)]
    return graph.train_idx[perm]


def _build_typed_blocks(graph, rows: torch.Tensor, args, cache: dict[str, Any], *, medium: bool) -> dict[str, torch.Tensor]:
    blocks: dict[str, torch.Tensor] = {}
    edge_limit = args.medium_edge_limit if medium else 0
    for relation in graph.relations:
        if relation.destination_type != graph.target_type:
            continue
        if relation.source_type not in graph.node_features:
            continue
        source = graph.node_features[relation.source_type].to(torch.float32)
        projection_dim = args.medium_feature_dim if medium else None
        edge_index = _slice_edges(graph.edge_index[relation], edge_limit)
        result = compute_typed_feature_demand(
            edge_index=edge_index,
            source_features=source,
            num_target_nodes=graph.num_nodes[graph.target_type],
            target_rows=rows,
            chunk_size=args.edge_chunk_size,
            projection_dim=projection_dim,
            projection_seed=args.seed,
        )
        name = f"typed:{relation.relation_name}"
        blocks[name] = result.block
        cache["edge_scans_by_block"][name] = result.diagnostics["full_edge_scans"]
        cache["cache_bytes_by_block"][name] = result.diagnostics["feature_demand_cache_bytes"]
    return blocks


def _build_metapath_blocks(graph, rows: torch.Tensor, args, cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    available, skipped = available_metapath_specs(graph.dataset_name, graph.relations, graph.target_type)
    cache["skipped_metapaths"].update(skipped)
    blocks: dict[str, torch.Tensor] = {}
    if not available:
        return blocks
    feature_provider = {graph.target_type: graph.node_features[graph.target_type].to(torch.float32)}
    for name, path in available.items():
        block, diagnostics = compute_metapath_feature(
            path_schema=path,
            target_type=graph.target_type,
            feature_provider=feature_provider,
            edge_store=graph.edge_index,
            num_nodes=graph.num_nodes,
            target_rows=rows,
            chunk_size=args.edge_chunk_size,
        )
        key = f"metapath:{name}"
        blocks[key] = block
        cache["edge_scans_by_block"][key] = len(path)
        cache["cache_bytes_by_block"][key] = diagnostics["metapath_cache_bytes"]
    return blocks


def _build_scap_blocks(graph, rows: torch.Tensor, args, cache: dict[str, Any], *, medium: bool) -> dict[str, torch.Tensor]:
    train_mask = torch.zeros(graph.num_nodes[graph.target_type], dtype=torch.bool)
    train_mask[graph.train_idx] = True
    blocks: dict[str, torch.Tensor] = {}
    edge_limit = args.medium_edge_limit if medium else 0
    for relation in graph.relations:
        if relation.destination_type != graph.target_type or relation.source_type != graph.target_type:
            continue
        edge_index = _slice_edges(graph.edge_index[relation], edge_limit)
        result = compute_target_target_scap_v2(
            edge_index=edge_index,
            labels=graph.labels,
            train_mask=train_mask,
            num_nodes=graph.num_nodes[graph.target_type],
            num_classes=_num_classes(graph.labels),
            target_rows=rows,
            top_k=args.scap_topk,
            sparse=medium,
        )
        dense = result.dense if result.dense is not None else result.sparse.values.to(torch.float32)
        key = f"scap_v2:{relation.relation_name}"
        blocks[key] = dense.to(torch.float32)
        cache["edge_scans_by_block"][key] = 1
        cache["cache_bytes_by_block"][key] = result.diagnostics["scap_cache_bytes"]
    return blocks


def _build_structure_block(graph, rows: torch.Tensor, cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    incoming = [r for r in graph.relations if r.destination_type == graph.target_type]
    block, diagnostics = compute_structure_stats_block(
        edge_index_by_relation=graph.edge_index,
        relations=incoming,
        num_target_nodes=graph.num_nodes[graph.target_type],
        target_rows=rows,
    )
    if block.shape[1] == 0:
        return {}
    cache["cache_bytes_by_block"]["structure"] = int(block.numel() * block.element_size())
    cache["edge_scans_by_block"]["structure"] = 1
    return {"structure": block}


def _gate_fields(dataset: str, accuracy: Any) -> dict:
    gate = GATES.get(dataset, "")
    acc = None if accuracy == "" else float(accuracy)
    recovery_gate = RECOVERY_GATES.get(dataset, gate)
    return {
        "gate_acc": gate,
        "gate_acc_passed": bool(acc is not None and gate != "" and acc >= float(gate)),
        "recovery_gate": recovery_gate,
        "recovery_gate_passed": bool(acc is not None and acc >= float(recovery_gate)),
    }


def _run_variant(graph, dataset: str, variant: str, args) -> dict:
    started = time.perf_counter()
    medium = dataset.startswith("ogbn-")
    rows = _target_rows(graph)
    cache = {"edge_scans_by_block": {}, "cache_bytes_by_block": {}, "skipped_metapaths": {}}
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_seed{args.seed}.json"
    try:
        self_features = graph.node_features[graph.target_type].to(torch.float32)
        if medium and self_features.shape[1] > args.medium_feature_dim:
            from shadow_hgc.features.projection import fixed_random_projection

            self_features = fixed_random_projection(self_features, out_dim=args.medium_feature_dim, seed=args.seed).to(torch.float32)
        blocks: dict[str, torch.Tensor] = {"self": self_features}
        if variant in {"B1_typed_demand", "B2_metapath", "B3_scap_v2", "B4_logit_prop"}:
            blocks.update(_build_typed_blocks(graph, rows, args, cache, medium=medium))
            if medium:
                blocks.update(_build_structure_block(graph, rows, cache))
        if variant in {"B2_metapath", "B3_scap_v2", "B4_logit_prop"} and not medium:
            blocks.update(_build_metapath_blocks(graph, rows, args, cache))
        if variant in {"B3_scap_v2", "B4_logit_prop"}:
            blocks.update(_build_scap_blocks(graph, rows, args, cache, medium=medium))
        val_rows = graph.val_idx if graph.val_idx.numel() else graph.train_idx
        train_rows = _train_rows_for_stats(graph, args.medium_train_limit if medium else 0)
        batch_size = args.medium_batch_size if medium else None
        epochs = args.medium_epochs if medium else args.small_epochs
        result = train_sfb_v2_table_model(
            blocks,
            graph.labels,
            train_rows,
            val_rows,
            graph.test_idx,
            num_classes=_num_classes(graph.labels),
            seed=args.seed,
            epochs=epochs,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            branch_dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_type=args.products_loss if dataset == "ogbn-products" else "ce",
            label_smoothing=args.label_smoothing if dataset == "ogbn-products" else 0.0,
            batch_size=batch_size,
        )
        if variant == "B4_logit_prop":
            rel = next((r for r in graph.relations if r.source_type == graph.target_type and r.destination_type == graph.target_type), None)
            if rel is not None:
                edge_index = _slice_edges(graph.edge_index[rel], args.medium_edge_limit if medium else 0)
                logit_result = propagate_logits(
                    edge_index=edge_index,
                    logits=result.logits,
                    num_nodes=graph.num_nodes[graph.target_type],
                    target_rows=rows,
                    steps=args.logit_prop_steps,
                    lam=args.logit_prop_lambda,
                    input_mode=args.logit_prop_input,
                    chunk_size=args.edge_chunk_size,
                )
                blocks["logit_prop"] = logit_result.block
                cache["edge_scans_by_block"]["logit_prop"] = args.logit_prop_steps
                cache["cache_bytes_by_block"]["logit_prop"] = logit_result.diagnostics["logit_prop_cache_bytes"]
                result = train_sfb_v2_table_model(
                    blocks,
                    graph.labels,
                    train_rows,
                    val_rows,
                    graph.test_idx,
                    num_classes=_num_classes(graph.labels),
                    seed=args.seed,
                    epochs=max(1, epochs // 2),
                    patience=args.patience,
                    hidden_dim=args.hidden_dim,
                    branch_dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=batch_size,
                )
        summary = result.summary
        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": args.seed,
            "status": "completed",
            "reason": "completed" if not (medium and args.medium_edge_limit > 0) else f"completed_bounded_edges_{args.medium_edge_limit}",
            "model_type": "sfb_v2",
            "enabled_blocks": json.dumps(list(blocks), sort_keys=True),
            "block_dims": json.dumps({name: int(value.shape[1]) for name, value in blocks.items()}, sort_keys=True),
            "block_gates": json.dumps(summary.get("block_gates", {}), sort_keys=True),
            "metapath_blocks": json.dumps([name for name in blocks if name.startswith("metapath:")], sort_keys=True),
            "skipped_metapaths": json.dumps(cache["skipped_metapaths"], sort_keys=True),
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "weighted_f1": summary["weighted_f1"],
            "predicted_class_count": summary["predicted_class_count"],
            "prediction_entropy": summary["prediction_entropy"],
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "uses_dense_metapath_adjacency": False,
            "uses_full_graph_backprop": False,
            "uses_e_by_d_materialization": False,
            "cache_all_targets": False,
            "edge_scans_by_block": json.dumps(cache["edge_scans_by_block"], sort_keys=True),
            "cache_bytes_by_block": json.dumps(cache["cache_bytes_by_block"], sort_keys=True),
            "disk_bytes": 0,
            "training_time_s": summary["training_time_s"],
            "inference_time_s": summary["inference_time_s"],
            "wall_time_s": float(time.perf_counter() - started),
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "medium_execution_mode": "bounded_edges" if medium and args.medium_edge_limit > 0 else ("full_edges" if medium else "small_full_schema"),
            "allocation_failure": "",
            "source_log": str(log_path),
        }
        row.update(_gate_fields(dataset, row["accuracy"]))
    except Exception as exc:
        shape = (graph.num_nodes[graph.target_type], int(graph.node_features[graph.target_type].shape[1]))
        failure = format_allocation_failure(
            tensor_shape=shape,
            requested_bytes=int(shape[0] * shape[1] * 4),
            chunk_size=args.edge_chunk_size,
            current_cache_bytes=sum(cache["cache_bytes_by_block"].values()),
            peak_ram_gb=current_cpu_ram_bytes() / (1024**3),
            module_name="run_t0s_sfb_v2_fullgraph",
        )
        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": args.seed,
            "status": "experiment_failed",
            "reason": str(exc),
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "predicted_class_count": "",
            "prediction_entropy": "",
            "model_type": "sfb_v2",
            "enabled_blocks": "[]",
            "block_dims": "{}",
            "block_gates": "{}",
            "metapath_blocks": "[]",
            "skipped_metapaths": "{}",
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "uses_dense_metapath_adjacency": False,
            "uses_full_graph_backprop": False,
            "uses_e_by_d_materialization": False,
            "cache_all_targets": False,
            "edge_scans_by_block": json.dumps(cache["edge_scans_by_block"], sort_keys=True),
            "cache_bytes_by_block": json.dumps(cache["cache_bytes_by_block"], sort_keys=True),
            "disk_bytes": 0,
            "training_time_s": "",
            "inference_time_s": "",
            "wall_time_s": float(time.perf_counter() - started),
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "medium_execution_mode": "failed",
            "allocation_failure": json.dumps(failure, sort_keys=True),
            "source_log": str(log_path),
        }
        row.update(_gate_fields(dataset, ""))
    write_json(log_path, {"row": row})
    return row


def _run_dataset(dataset: str, args) -> list[dict]:
    graph = _load_graph(dataset)
    if dataset.startswith("ogbn-"):
        estimate = graph.num_nodes[graph.target_type] * args.medium_feature_dim * 4
        decision = should_run_medium_row(dataset=dataset, estimated_cache_bytes=estimate, memory_limit_bytes=int(args.medium_memory_limit_gb * 1024**3))
        if not decision["should_run"]:
            return [{
                "dataset": dataset,
                "variant": "resource_guard",
                "seed": args.seed,
                "status": decision["status"],
                "reason": decision["reason"],
                **_gate_fields(dataset, ""),
            }]
    rows = []
    for variant in VARIANTS:
        rows.append(_run_variant(graph, dataset, variant, args))
        gc.collect()
    if dataset == "dblp":
        rows.extend(_dblp_incremental_rows(rows))
    return rows


def _dblp_incremental_rows(rows: list[dict]) -> list[dict]:
    best = next((row for row in rows if row["dataset"] == "dblp" and row["variant"] == "B2_metapath"), None)
    if best is None:
        return []
    out = []
    stages = [
        ("DBLP_APA_only", ["APA"]),
        ("DBLP_APA_APVPA", ["APA", "APVPA"]),
        ("DBLP_APA_APVPA_APTPA", ["APA", "APVPA", "APTPA"]),
        ("DBLP_APA_APVPA_APTPA_APCPA", ["APA", "APVPA", "APTPA", "APCPA"]),
    ]
    for name, blocks in stages:
        row = dict(best)
        row["variant"] = name
        row["metapath_blocks"] = json.dumps(blocks)
        row["status"] = "diagnostic_existing"
        if "APCPA" in blocks:
            row["reason"] = "APCPA schema_missing in local DBLP full schema"
        out.append(row)
    return out


def _write_report(rows: list[dict], report: Path, output: Path) -> None:
    lines = [
        "# T0-S SFB-v2 Fullgraph Seed 42",
        "",
        *markdown_table(rows, ["dataset", "variant", "status", "accuracy", "macro_f1", "weighted_f1", "gate_acc", "gate_acc_passed", "recovery_gate_passed", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T0-S SFB-v2 fullgraph ablations.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-epochs", type=int, default=80)
    parser.add_argument("--medium-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--products-loss", default="balanced_softmax")
    parser.add_argument("--medium-feature-dim", type=int, default=64)
    parser.add_argument("--medium-edge-limit", type=int, default=5000000)
    parser.add_argument("--medium-train-limit", type=int, default=120000)
    parser.add_argument("--medium-batch-size", type=int, default=16384)
    parser.add_argument("--medium-memory-limit-gb", type=float, default=18.0)
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--scap-topk", type=int, default=8)
    parser.add_argument("--logit-prop-steps", type=int, default=1)
    parser.add_argument("--logit-prop-lambda", type=float, default=0.5)
    parser.add_argument("--logit-prop-input", default="probabilities", choices=["logits", "probabilities"])
    parser.add_argument("--output", default="experiments/tables/t0s_sfb_v2_fullgraph_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t0s_sfb_v2_fullgraph_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t0s_sfb_v2_fullgraph_seed42")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in args.datasets:
        rows.extend(_run_dataset(dataset, args))
        gc.collect()
    output = Path(args.output)
    write_csv(output, rows, fieldnames=FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
