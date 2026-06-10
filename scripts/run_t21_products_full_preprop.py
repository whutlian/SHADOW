from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import load_t2_graph, num_classes, split_train_valid
from scripts.t21_common import T21_PRODUCTS_FIELDS, markdown_table, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.preprop.true_preprop import compute_preprop_blocks
from shadow_hgc.train.train_sft_teacher import train_sft_teacher
from shadow_hgc.train.lazy_sft_memmap import load_products_labels_and_splits, train_lazy_sft_from_memmap


def _read_memmap(path: Path, shape: list[int], dtype: str) -> torch.Tensor:
    array = np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=tuple(shape))
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32)


def _blocked_row(args, reason: str) -> dict[str, Any]:
    return {
        "dataset": "ogbn-products",
        "target_type": "product",
        "status": "blocked_requires_explicit_full_run",
        "reason": reason,
        "run_mode": "not_run",
        "selected_blocks": "[]",
        "preprop_blocks": "[]",
        "manifest_dir": "",
        "total_cache_bytes": 0,
        "full_edge_scans": 0,
        "edge_chunk_size": args.edge_chunk_size,
        "dst_chunk_size": args.dst_chunk_size,
        "feature_dim": args.feature_dim,
        "training_epochs": 0,
        "training_time_s": 0,
        "inference_time_s": 0,
        "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "uses_diffusion_legacy": False,
    }


def run_products(args) -> dict[str, Any]:
    if not args.run_full:
        return _blocked_row(args, "full products run requires --run-full to avoid accidental local OOM")
    started = time.perf_counter()
    try:
        if args.lazy_sft:
            manifest_dir = Path(args.output_dir)
            manifest_path = manifest_dir / "manifest.json"
            if not manifest_path.exists():
                return {
                    **_blocked_row(args, f"lazy SFT requires existing manifest at {manifest_path}; run full preprop first"),
                    "status": "blocked_missing_manifest",
                    "run_mode": "lazy_memmap_gpu",
                }
            labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.dataset_root)
            device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
            lazy_result = train_lazy_sft_from_memmap(
                manifest_dir=manifest_dir,
                labels=labels,
                train_rows=train_rows,
                valid_rows=valid_rows,
                test_rows=test_rows,
                num_classes=int(labels.max().item()) + 1,
                device=device,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                loss_type=args.loss_type,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.train_epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed,
                label_smoothing=args.label_smoothing,
            )
            summary = lazy_result.summary
            log_path = Path(args.log_dir) / f"ogbn_products_lazy_sft_seed{args.seed}.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            test_metrics = summary["test"]
            return {
                "dataset": "ogbn-products",
                "target_type": "product",
                "status": "completed",
                "reason": "lazy_memmap_gpu_sft_completed" if str(device).startswith("cuda") else "lazy_memmap_cpu_sft_completed",
                "run_mode": f"lazy_memmap_{device}",
                "accuracy": test_metrics.get("accuracy", ""),
                "macro_f1": test_metrics.get("macro_f1", ""),
                "predicted_class_count": test_metrics.get("predicted_class_count", ""),
                "selected_blocks": json.dumps(list(summary.get("block_dims", {}).keys()), sort_keys=True),
                "preprop_blocks": json.dumps([block["name"] for block in manifest.get("blocks", [])], sort_keys=True),
                "manifest_dir": str(manifest_dir),
                "total_cache_bytes": manifest.get("total_cache_bytes", ""),
                "full_edge_scans": manifest.get("full_edge_scans", ""),
                "edge_chunk_size": manifest.get("edge_chunk_size", args.edge_chunk_size),
                "dst_chunk_size": manifest.get("dst_chunk_size", args.dst_chunk_size),
                "feature_dim": manifest.get("feature_dim", args.feature_dim),
                "training_epochs": args.train_epochs,
                "training_time_s": summary.get("training_time_s", ""),
                "inference_time_s": summary.get("inference_time_s", ""),
                "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", current_cpu_ram_bytes() / (1024**3)),
                "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", current_gpu_ram_bytes() / (1024**3)),
                "uses_logits_as_input": False,
                "uses_teacher_logits": False,
                "uses_kd": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_e_by_d_materialization": False,
                "uses_diffusion_legacy": False,
            }
        graph = load_t2_graph("ogbn-products")
        train_rows, valid_rows = split_train_valid(graph, seed=args.seed)
        provider = {name: value.to(torch.float32) for name, value in graph.node_features.items()}
        provider["train_rows"] = train_rows
        blocks = ["X0", *[f"X1_{rel.relation_name}" for rel in graph.relations if rel.source_type == graph.target_type and rel.destination_type == graph.target_type]]
        manifest_dir = Path(args.output_dir)
        manifest = compute_preprop_blocks(
            dataset_name="ogbn-products",
            target_type=graph.target_type,
            x_provider=provider,
            relations=graph.edge_index,
            output_dir=str(manifest_dir),
            blocks=blocks,
            feature_dim=args.feature_dim,
            dtype=args.dtype,
            edge_chunk_size=args.edge_chunk_size,
            dst_chunk_size=args.dst_chunk_size,
            force_memmap=True,
            seed=args.seed,
        )
        metrics: dict[str, Any] = {}
        training_time = 0.0
        inference_time = 0.0
        selected_blocks: list[str] = []
        if args.train_epochs > 0:
            block_map = {}
            for meta in manifest.blocks:
                if meta.name == "X0":
                    key = "self"
                elif meta.name.startswith("X1_"):
                    key = meta.name.lower()
                else:
                    continue
                block_map[key] = _read_memmap(manifest_dir / meta.path, meta.shape, meta.dtype)
            selected_blocks = list(block_map)
            train_result = train_sft_teacher(
                block_map,
                graph.labels,
                train_rows,
                valid_rows,
                graph.test_idx,
                num_classes=num_classes(graph.labels),
                model_type="sagn_lite",
                hidden_dim=args.hidden_dim,
                dropout=0.3,
                loss_type=args.loss_type,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.train_epochs,
                patience=max(1, args.train_epochs),
                seed=args.seed,
                batch_size=args.batch_size,
                label_smoothing=args.label_smoothing,
            )
            metrics = train_result.summary["test"]
            training_time = float(train_result.summary["training_time_s"])
        status = "completed" if metrics else "preprop_completed"
        return {
            "dataset": "ogbn-products",
            "target_type": "product",
            "status": status,
            "reason": "full_edge_products_preprop_completed" if metrics == {} else "full_edge_products_preprop_and_sft_completed",
            "run_mode": "full_edges",
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "predicted_class_count": metrics.get("predicted_class_count", ""),
            "selected_blocks": json.dumps(selected_blocks, sort_keys=True),
            "preprop_blocks": json.dumps([block.name for block in manifest.blocks], sort_keys=True),
            "manifest_dir": str(manifest_dir),
            "total_cache_bytes": manifest.total_cache_bytes,
            "full_edge_scans": manifest.full_edge_scans,
            "edge_chunk_size": manifest.edge_chunk_size,
            "dst_chunk_size": manifest.dst_chunk_size,
            "feature_dim": manifest.block_dim,
            "training_epochs": args.train_epochs,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "uses_logits_as_input": False,
            "uses_teacher_logits": False,
            "uses_kd": False,
            "uses_dense_p2": False,
            "uses_bounded_edges": False,
            "uses_e_by_d_materialization": False,
            "uses_diffusion_legacy": False,
        }
    except Exception as exc:
        return {
            **_blocked_row(args, f"full products execution failed: {type(exc).__name__}: {exc}"),
            "status": "blocked_full_execution_failed",
            "run_mode": "full_edges_attempted",
            "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
            "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
            "training_time_s": float(time.perf_counter() - started),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2.1 ogbn-products full-edge preprop execution.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--lazy-sft", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dataset-root", default="dataset/ogbn_products")
    parser.add_argument("--model-type", default="gamlp_lite", choices=["sagn_lite", "gamlp_lite", "residual_block_gated"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output-dir", default="experiments/preprop/t21_products_seed42")
    parser.add_argument("--log-dir", default="experiments/logs/t21_products_lazy_sft_seed42")
    parser.add_argument("--output", default="experiments/tables/t21_products_full_execution_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_products_full_execution_summary.md")
    args = parser.parse_args()
    rows = [run_products(args)]
    output = write_csv(args.output, rows, T21_PRODUCTS_FIELDS)
    lines = [
        "# T2.1 ogbn-products Full Execution",
        "",
        "This row never uses bounded edges, logits, KD, dense P2, legacy diffusion, or E x d materialization. If `--run-full` is not supplied, the row is explicitly blocked rather than promoted.",
        "",
        *markdown_table(rows, ["dataset", "status", "run_mode", "accuracy", "macro_f1", "full_edge_scans", "total_cache_bytes", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output), "status": rows[0]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
