from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.schema_audit import feature_hash, label_hash, schema_hash, split_hash
from shadow_hgc.data.small import load_processed_small_dataset, load_processed_small_dataset_full_schema
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.features.scap_blocks import build_scap_blocks_for_graph
from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv, write_json
from shadow_hgc.fullgraph.sfb_train import train_sfb_table_model
from shadow_hgc.fullgraph.t0s_gates import T0S_ACCURACY_GATES, evaluate_t0s_row


FIELDNAMES = [
    "dataset",
    "variant",
    "seed",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "predicted_class_count",
    "gate_acc",
    "gate_acc_passed",
    "gate_scalability_passed",
    "gate_passed",
    "blocked_reason",
    "model_type",
    "feature_blocks",
    "scap_blocks",
    "path_scap_blocks",
    "uses_diffusion",
    "uses_dense_p2",
    "uses_full_graph_backprop",
    "train_label_only",
    "target_type",
    "num_classes",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "num_nodes_by_type",
    "num_edges_by_type",
    "split_hash",
    "feature_hash",
    "label_hash",
    "schema_hash",
    "training_time_s",
    "inference_time_s",
    "wall_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "cache_bytes",
    "full_edge_scans",
    "scap_cache_metadata",
    "sfb_diagnostics",
    "source_log",
]


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return int(valid.max().item()) + 1 if valid.numel() else 0


def _graph_counts(graph) -> tuple[str, str]:
    node_counts = json.dumps({key: int(value) for key, value in sorted(graph.num_nodes.items())}, sort_keys=True)
    edge_counts = json.dumps({str(rel): int(graph.edge_index[rel].shape[1]) for rel in graph.relations}, sort_keys=True)
    return node_counts, edge_counts


def _base_row(graph, *, dataset: str, variant: str, seed: int, status: str, reason: str) -> dict[str, Any]:
    node_counts, edge_counts = _graph_counts(graph)
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "status": status,
        "reason": reason,
        "model_type": "sfb",
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_full_graph_backprop": False,
        "train_label_only": True,
        "target_type": graph.target_type,
        "num_classes": _num_classes(graph.labels),
        "train_nodes": int(graph.train_idx.numel()),
        "valid_nodes": int(graph.val_idx.numel()),
        "test_nodes": int(graph.test_idx.numel()),
        "num_nodes_by_type": node_counts,
        "num_edges_by_type": edge_counts,
        "split_hash": split_hash(graph),
        "feature_hash": feature_hash(graph),
        "label_hash": label_hash(graph),
        "schema_hash": schema_hash(graph),
        "path_scap_blocks": "[]",
    }


def _run_small_dataset(dataset: str, args: argparse.Namespace, *, full_schema: bool, hidden_dim: int, dropout: float, lr: float) -> dict[str, Any]:
    graph = load_processed_small_dataset_full_schema(dataset) if full_schema else load_processed_small_dataset(dataset)
    variant = "t0s_sfb_scap_full_schema" if full_schema else "t0s_sfb_scap"
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        summary = json.loads(log_path.read_text(encoding="utf-8"))
        row = dict(summary["row"])
        row["source_log"] = str(log_path)
        return row

    start = time.perf_counter()
    try:
        target_features = graph.node_features[graph.target_type].to(torch.float32)
        scap_result = build_scap_blocks_for_graph(
            graph,
            prior_centering=True,
            log1p=True,
            l2_normalize=False,
            hub_cap=args.hub_cap,
        )
        blocks = {"self": target_features, **{name: block.to(torch.float32) for name, block in scap_result.blocks.items()}}
        val_rows = graph.val_idx if graph.val_idx.numel() else graph.train_idx
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        run = train_sfb_table_model(
            blocks,
            graph.labels,
            graph.train_idx,
            val_rows,
            graph.test_idx,
            num_classes=_num_classes(graph.labels),
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            dropout=dropout,
            block_dropout=args.block_dropout,
            lr=lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            fusion="residual_logits",
        )
        cache_bytes = int(sum(meta.get("cache_bytes", 0) for meta in scap_result.diagnostics.values()))
        full_edge_scans = int(max([meta.get("full_edge_scans", 0) for meta in scap_result.diagnostics.values()] or [0]))
        scap_block_names = [name for name in scap_result.blocks if not name.startswith("path_scap:")]
        path_scap_block_names = [name for name in scap_result.blocks if name.startswith("path_scap:")]
        row = {
            **_base_row(graph, dataset=dataset, variant=variant, seed=args.seed, status="completed", reason="completed"),
            "accuracy": run.summary.get("accuracy", ""),
            "macro_f1": run.summary.get("macro_f1", ""),
            "weighted_f1": run.summary.get("weighted_f1", ""),
            "predicted_class_count": run.summary.get("predicted_class_count", ""),
            "feature_blocks": json.dumps(list(blocks), sort_keys=True),
            "scap_blocks": json.dumps(scap_block_names, sort_keys=True),
            "path_scap_blocks": json.dumps(path_scap_block_names, sort_keys=True),
            "training_time_s": run.summary.get("training_time_s", ""),
            "inference_time_s": 0.0,
            "wall_time_s": float(time.perf_counter() - start),
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "cache_bytes": cache_bytes,
            "full_edge_scans": full_edge_scans,
            "scap_cache_metadata": json.dumps(scap_result.diagnostics, sort_keys=True),
            "sfb_diagnostics": json.dumps(run.model.diagnostics(), sort_keys=True),
            "source_log": str(log_path),
        }
    except Exception as exc:
        row = {
            **_base_row(graph, dataset=dataset, variant=variant, seed=args.seed, status="experiment_failed", reason=str(exc)),
            "uses_diffusion": False,
            "uses_dense_p2": False,
            "uses_full_graph_backprop": False,
            "train_label_only": True,
            "feature_blocks": "[]",
            "scap_blocks": "[]",
            "accuracy": "",
            "wall_time_s": float(time.perf_counter() - start),
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "cache_bytes": 0,
            "full_edge_scans": 0,
            "source_log": str(log_path),
        }
    row = evaluate_t0s_row(row)
    write_json(log_path, {"row": row})
    return row


def _medium_guard_row(dataset: str, seed: int) -> dict[str, Any]:
    target_type = "paper" if dataset == "ogbn-arxiv" else "product"
    reason = (
        "medium fullgraph SFB+SCAP is resource-guarded on this local desktop; "
        "no diffusion, dense P2, or full-graph backprop path was executed"
    )
    row = {
        "dataset": dataset,
        "variant": "t0s_sfb_scap_resource_guard",
        "seed": seed,
        "status": "skipped_resource_guard",
        "reason": reason,
        "accuracy": "",
        "macro_f1": "",
        "weighted_f1": "",
        "predicted_class_count": "",
        "model_type": "sfb",
        "feature_blocks": json.dumps(["self", "scap:incoming_target_relations"]),
        "scap_blocks": json.dumps(["resource_guarded"]),
        "path_scap_blocks": "[]",
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_full_graph_backprop": False,
        "train_label_only": True,
        "target_type": target_type,
        "num_classes": "",
        "train_nodes": "",
        "valid_nodes": "",
        "test_nodes": "",
        "num_nodes_by_type": "{}",
        "num_edges_by_type": "{}",
        "split_hash": "not_loaded_resource_guard",
        "feature_hash": "not_loaded_resource_guard",
        "label_hash": "not_loaded_resource_guard",
        "schema_hash": "not_loaded_resource_guard",
        "training_time_s": "",
        "inference_time_s": "",
        "wall_time_s": 0.0,
        "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
        "cache_bytes": 0,
        "full_edge_scans": 2,
        "scap_cache_metadata": json.dumps({"resource_guard": True, "cache_all_targets": False}),
        "sfb_diagnostics": "{}",
        "source_log": "",
    }
    return evaluate_t0s_row(row)


def _write_report(rows: list[dict[str, Any]], report: Path, csv_path: Path) -> None:
    lines = [
        "# T0-S Fullgraph Parity Seed 42",
        "",
        "Promoted rows must pass both the dataset accuracy gate and the scalability gate.",
        "",
        *markdown_table(
            rows,
            ["dataset", "variant", "status", "accuracy", "gate_acc", "gate_acc_passed", "gate_scalability_passed", "blocked_reason"],
        ),
        "",
        "Scalability policy: no diffusion, no dense P2, no full-graph backprop, train-label-only SCAP, and no all-target demand cache.",
        "",
        f"- CSV: `{csv_path}`",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run opt-in T0-S fullgraph parity rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hub-cap", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/t0s_fullgraph_parity_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t0s_fullgraph_parity_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t0s_fullgraph_parity_seed42")
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = [
        _run_small_dataset("acm", args, full_schema=False, hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr),
        _run_small_dataset("dblp", args, full_schema=True, hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr),
        _run_small_dataset("imdb", args, full_schema=False, hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr),
        _medium_guard_row("ogbn-arxiv", args.seed),
        _medium_guard_row("ogbn-products", args.seed),
    ]
    output = Path(args.output)
    write_csv(output, rows, fieldnames=FIELDNAMES)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
