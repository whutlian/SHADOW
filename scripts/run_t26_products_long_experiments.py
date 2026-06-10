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

from scripts.run_t24_products_sft_recovery import _train_eval
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.products_recovery_t26 import mixed_class_budget, nearest_prototype_oracle
from shadow_hgc.sft.signature_cache import write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store, load_products_labels_and_splits


PRODUCTS_NODES = 2_449_029

FIELDS = [
    "dataset",
    "method",
    "seed",
    "requested_full_node_ratio",
    "target_prototypes",
    "shadow_nodes",
    "total_condensed_edges",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "status",
    "p0a_alltrain_acc",
    "p0b_self_fit_acc",
    "p0d_prototype_oracle_acc",
    "p0d_centroid_oracle_acc",
    "training_time",
    "inference_time",
    "condensation_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "cache_bytes",
    "notes",
]


def _load_train_signature(signature_dir: str | Path, metadata: dict[str, Any]) -> torch.Tensor:
    train_meta = metadata["arrays"]["train_signature"]
    array = np.memmap(
        Path(signature_dir) / train_meta["path"],
        mode="r",
        dtype=np.dtype(train_meta["dtype"]),
        shape=tuple(int(value) for value in train_meta["shape"]),
    )
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _budgeted_hybrid(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, *, seed: int) -> torch.Tensor:
    return select_classwise_coreset_rows(signature, labels, train_rows, total, mode="hybrid", seed=int(seed))


def _budgeted_random(signature: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor, total: int, *, seed: int) -> torch.Tensor:
    return select_classwise_coreset_rows(signature, labels, train_rows, total, mode="random", seed=int(seed))


def run_products_long(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.products_root)
    store = load_manifest_block_store(args.manifest_dir).subset(json.loads(args.selected_blocks))
    signature_cache = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=json.loads(args.selected_blocks),
        batch_size=int(args.signature_batch_size),
    )
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    rows: list[dict[str, Any]] = []
    num_classes = int(labels.max().item()) + 1
    common = {
        "dataset": "ogbn-products",
        "seed": int(args.seed),
        "shadow_nodes": 0,
        "peak_cpu_ram": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram": current_gpu_ram_bytes() / (1024**3),
        "cache_bytes": int(signature_cache.metadata["cache_bytes"]),
    }

    if bool(args.run_p0a):
        started = time.perf_counter()
        metrics, train_s, infer_s = _train_eval(
            store,
            labels,
            train_rows,
            valid_rows,
            test_rows,
            train_rows,
            epochs=int(args.p0a_epochs),
            hidden_dim=int(args.hidden_dim),
            device=args.device,
        )
        rows.append(
            {
                **common,
                "method": "P0a_alltrain_condensed_trainer_parity",
                "requested_full_node_ratio": float(train_rows.numel()) / PRODUCTS_NODES,
                "target_prototypes": int(train_rows.numel()),
                "total_condensed_edges": int(train_rows.numel()),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "predicted_class_count": int(metrics["predicted_class_count"]),
                "status": "completed_long",
                "p0a_alltrain_acc": float(metrics["accuracy"]),
                "training_time": train_s,
                "inference_time": infer_s,
                "condensation_time": time.perf_counter() - started,
                "notes": f"all-train condensed trainer parity, epochs={int(args.p0a_epochs)}",
            }
        )

    for ratio in [float(value) for value in args.ratios]:
        total = max(num_classes, int(round(PRODUCTS_NODES * ratio)))
        cond_started = time.perf_counter()
        selected = _budgeted_hybrid(signature, labels, train_rows, total, seed=int(args.seed))
        condensation_time = time.perf_counter() - cond_started
        if bool(args.run_p0b):
            metrics, train_s, infer_s = _train_eval(
                store,
                labels,
                train_rows,
                selected,
                selected,
                selected,
                epochs=int(args.p0b_epochs),
                hidden_dim=int(args.hidden_dim),
                device=args.device,
            )
            rows.append(
                {
                    **common,
                    "method": "P0b_selected_prototype_self_fit",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(selected.numel()),
                    "total_condensed_edges": int(selected.numel()),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "predicted_class_count": int(metrics["predicted_class_count"]),
                    "status": "completed_long",
                    "p0b_self_fit_acc": float(metrics["accuracy"]),
                    "training_time": train_s,
                    "inference_time": infer_s,
                    "condensation_time": condensation_time,
                    "notes": f"selected prototype self-fit, epochs={int(args.p0b_epochs)}",
                }
            )
        if bool(args.run_p0c):
            random_selected = _budgeted_random(signature, labels, train_rows, total, seed=int(args.seed))
            metrics, train_s, infer_s = _train_eval(
                store,
                labels,
                train_rows,
                valid_rows,
                test_rows,
                random_selected,
                epochs=int(args.p0c_epochs),
                hidden_dim=int(args.hidden_dim),
                device=args.device,
            )
            rows.append(
                {
                    **common,
                    "method": "P0c_same_budget_random_subset",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(random_selected.numel()),
                    "total_condensed_edges": int(random_selected.numel()),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "predicted_class_count": int(metrics["predicted_class_count"]),
                    "status": "completed_long",
                    "training_time": train_s,
                    "inference_time": infer_s,
                    "condensation_time": condensation_time,
                    "notes": f"same-budget random subset, epochs={int(args.p0c_epochs)}",
                }
            )
        if bool(args.run_p0d):
            selected_pos = torch.searchsorted(train_rows, selected)
            oracle = nearest_prototype_oracle(
                signature,
                labels[train_rows],
                selected_pos,
                signature,
                labels[train_rows],
                metric="euclidean",
            )
            rows.append(
                {
                    **common,
                    "method": "P0d_nearest_prototype_oracle",
                    "requested_full_node_ratio": ratio,
                    "target_prototypes": int(selected.numel()),
                    "total_condensed_edges": int(selected.numel()),
                    "accuracy": oracle["prototype_oracle_acc"],
                    "macro_f1": "",
                    "predicted_class_count": "",
                    "status": "completed_long_train_signature_oracle",
                    "p0d_prototype_oracle_acc": oracle["prototype_oracle_acc"],
                    "p0d_centroid_oracle_acc": oracle["centroid_oracle_acc"],
                    "condensation_time": condensation_time,
                    "notes": "nearest prototype oracle on train SFT signatures; no valid/test labels used for selection",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real T26 products long diagnostics.")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Xres2","structure","Y1","Y2","Y3"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/ogbn-products/t26_long")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--run-p0a", action="store_true", default=True)
    parser.add_argument("--run-p0b", action="store_true", default=True)
    parser.add_argument("--run-p0c", action="store_true", default=True)
    parser.add_argument("--run-p0d", action="store_true")
    parser.add_argument("--p0a-epochs", type=int, default=20)
    parser.add_argument("--p0b-epochs", type=int, default=80)
    parser.add_argument("--p0c-epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t26_products_long_experiments_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_products_long_experiments.md")
    args = parser.parse_args()
    rows = run_products_long(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Products Long Experiments",
            "",
            f"- Device: `{args.device}`",
            f"- P0a epochs: `{int(args.p0a_epochs)}`",
            f"- P0b epochs: `{int(args.p0b_epochs)}`",
            f"- P0c epochs: `{int(args.p0c_epochs)}`",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "status", "accuracy", "macro_f1", "predicted_class_count", "training_time", "inference_time", "notes"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
