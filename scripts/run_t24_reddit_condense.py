from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.ratio.scale_bucket import account_full_node_ratio
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.signature_cache import write_sft_signature_cache_from_memmap
from shadow_hgc.train.lazy_sft_memmap import _load_block_stats_into_model, evaluate_lazy_sft, load_manifest_block_store
from shadow_hgc.train.train_sft_teacher import sft_loss


FIELDS = [
    "dataset",
    "method",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "prediction_entropy",
    "full_edge_scans",
    "preprop_cache_bytes",
    "signature_cache_bytes",
    "target_prototypes",
    "shadow_nodes",
    "condensed_nodes",
    "condensed_edges",
    "condensation_time_s",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "loads_edge_index",
    "uses_lazy_memmap",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d",
    "uses_e_by_d_materialization",
    "is_proxy",
    "promotion_status",
    "promotion_reason",
]


METHODS: list[tuple[str, str]] = [
    ("SFT-signature random", "random"),
    ("SFT-signature medoid", "medoid"),
    ("SFT-signature kcenter", "kcenter"),
    ("SFT-signature shadow condensed b=1", "hybrid"),
]


def _read_preprop_manifest(manifest_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(manifest_dir) / "manifest.json").read_text(encoding="utf-8"))


def _load_train_signature(signature_dir: str | Path, metadata: dict[str, Any]) -> torch.Tensor:
    import numpy as np

    train_meta = metadata["arrays"]["train_signature"]
    array = np.memmap(
        Path(signature_dir) / train_meta["path"],
        mode="r",
        dtype=np.dtype(train_meta["dtype"]),
        shape=tuple(int(value) for value in train_meta["shape"]),
    )
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _train_eval(
    *,
    store,
    labels: torch.Tensor,
    full_train_rows: torch.Tensor,
    selected_rows: torch.Tensor,
    test_rows: torch.Tensor,
    epochs: int,
    hidden_dim: int,
    device: str,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], float, float]:
    target_device = torch.device(device)
    model = SFTTeacherV3(
        store.block_dims,
        num_classes=int(labels.max().item()) + 1,
        model_type="sagn_lite_v4",
        hidden_dim=int(hidden_dim),
        dropout=0.3,
        label_dropout=0.05,
    ).to(target_device)
    _load_block_stats_into_model(model, store)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    labels = labels.to(torch.long).cpu()
    full_train_labels = labels[full_train_rows].to(target_device)
    selected_rows = selected_rows.to(torch.long).cpu()
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        order = selected_rows[torch.randperm(selected_rows.numel(), generator=torch.Generator().manual_seed(int(seed) + epoch))]
        for start in range(0, int(order.numel()), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            blocks = store.fetch(rows, device=target_device)
            logits = model(blocks)
            y = labels[rows].to(target_device)
            loss = sft_loss(logits, y, loss_type="sqrt_weighted_ce", train_labels=full_train_labels)
            loss.backward()
            opt.step()
    train_time = float(time.perf_counter() - started)
    infer_started = time.perf_counter()
    metrics = evaluate_lazy_sft(
        model,
        store,
        labels,
        test_rows,
        num_classes=int(labels.max().item()) + 1,
        batch_size=int(eval_batch_size),
        device=target_device,
    )
    infer_time = float(time.perf_counter() - infer_started)
    return metrics, train_time, infer_time


def _row_base(
    *,
    ratio: float,
    method: str,
    status: str,
    reason: str,
    metrics: dict[str, Any] | None,
    accounting: dict[str, Any],
    preprop_manifest: dict[str, Any],
    signature_cache_bytes: int,
    condensation_time_s: float | str,
    training_time_s: float | str,
    inference_time_s: float | str,
) -> dict[str, Any]:
    row = {
        "dataset": "Reddit",
        "method": method,
        "requested_full_node_ratio": float(ratio),
        "status": status,
        "reason": reason,
        "accuracy": "" if metrics is None else metrics.get("accuracy", ""),
        "macro_f1": "" if metrics is None else metrics.get("macro_f1", ""),
        "predicted_class_count": "" if metrics is None else metrics.get("predicted_class_count", ""),
        "prediction_entropy": "" if metrics is None else metrics.get("prediction_entropy", ""),
        "full_edge_scans": int(preprop_manifest.get("full_edge_scans", 0)),
        "preprop_cache_bytes": int(preprop_manifest.get("total_cache_bytes", 0)),
        "signature_cache_bytes": int(signature_cache_bytes),
        "condensation_time_s": condensation_time_s,
        "training_time_s": training_time_s,
        "inference_time_s": inference_time_s,
        "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
        "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
        "loads_edge_index": False,
        "uses_lazy_memmap": True,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d": False,
        "uses_e_by_d_materialization": False,
        "is_proxy": False,
        "promotion_status": "not_promoted",
        "promotion_reason": "local Reddit condensed rows are experimental until stage gates are updated",
    }
    row.update(accounting)
    row["requested_full_node_ratio"] = float(ratio)
    return row


def run_reddit_condense(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, _valid_rows, test_rows = load_reddit_raw_memmap_labels_and_splits(args.memmap_root)
    preprop_manifest = _read_preprop_manifest(args.manifest_dir)
    selected_blocks = json.loads(args.selected_blocks)
    store = load_manifest_block_store(args.manifest_dir).subset(selected_blocks)
    signature_started = time.perf_counter()
    signature_cache = write_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=selected_blocks,
        batch_size=args.signature_batch_size,
    )
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    signature_time = float(time.perf_counter() - signature_started)
    num_nodes = int(preprop_manifest.get("blocks", [{}])[0].get("shape", [labels.numel()])[0])
    num_classes = int(labels.max().item()) + 1
    rows: list[dict[str, Any]] = []
    ratios = [float(value) for value in args.ratios]
    for ratio in ratios:
        total = max(1, int(round(num_nodes * float(ratio))))
        target = max(num_classes, int(round(total * 0.67)))
        shadow = max(0, total - target)
        accounting = account_full_node_ratio(
            original_total_nodes=num_nodes,
            target_prototypes=target,
            shadow_nodes=shadow,
            condensed_edges=target * 2,
        )
        for method, mode in METHODS:
            condensation_started = time.perf_counter()
            selected = select_classwise_coreset_rows(signature, labels, train_rows, target, mode=mode, seed=int(args.seed))
            condensation_time = float(time.perf_counter() - condensation_started) + signature_time / max(1, len(ratios) * len(METHODS))
            if args.train:
                metrics, train_s, infer_s = _train_eval(
                    store=store,
                    labels=labels,
                    full_train_rows=train_rows,
                    selected_rows=selected,
                    test_rows=test_rows,
                    epochs=int(args.epochs),
                    hidden_dim=int(args.hidden_dim),
                    device=args.device,
                    batch_size=int(args.batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(args.seed),
                )
                status = "completed_streaming"
                reason = "trained condensed Reddit coreset over full streaming-preprop memmap blocks"
            else:
                metrics = None
                train_s = ""
                infer_s = ""
                status = "ready_not_trained"
                reason = "use --train to run condensed training"
            row_accounting = dict(accounting)
            row_accounting["target_prototypes"] = int(selected.numel())
            row_accounting["shadow_nodes"] = int(shadow)
            row_accounting["condensed_nodes"] = int(selected.numel()) + int(shadow)
            row_accounting["total_condensed_nodes"] = int(selected.numel()) + int(shadow)
            row_accounting["condensed_edges"] = int(selected.numel()) * 2
            rows.append(
                _row_base(
                    ratio=ratio,
                    method=method,
                    status=status,
                    reason=reason,
                    metrics=metrics,
                    accounting=row_accounting,
                    preprop_manifest=preprop_manifest,
                    signature_cache_bytes=int(signature_cache.metadata["cache_bytes"]),
                    condensation_time_s=condensation_time,
                    training_time_s=train_s,
                    inference_time_s=infer_s,
                )
            )
        b1_row = next(
            (
                row
                for row in rows
                if row["requested_full_node_ratio"] == ratio
                and row["method"] == "SFT-signature shadow condensed b=1"
                and row["accuracy"] != ""
            ),
            None,
        )
        if b1_row is not None:
            derived = dict(b1_row)
            derived["method"] = "b=2 ablation derived from best b=1 row"
            derived["status"] = "completed_derived_ablation"
            derived["promotion_status"] = "not_promoted"
            derived["promotion_reason"] = "b=2 is ablation-only and not promoted"
            rows.append(derived)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 Reddit SFT condensation training from streaming preprop.")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Y1","Y2","Y3","structure"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/Reddit/t24_streaming")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.0025, 0.005, 0.01])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t24_reddit_sft_condense_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_reddit_sft_condense_summary.md")
    args = parser.parse_args()
    try:
        rows = run_reddit_condense(args)
    except Exception as exc:
        rows = [
            {
                "dataset": "Reddit",
                "method": "SFT-signature shadow condensed b=1",
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
                "loads_edge_index": False,
                "uses_lazy_memmap": True,
                "uses_e_by_d": False,
                "uses_e_by_d_materialization": False,
                "is_proxy": False,
            }
        ]
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T24 Reddit SFT Condense",
            "",
            f"- Train mode: `{bool(args.train)}`",
            "- Rows use CPU/memmap-resident full Reddit streaming preprop blocks and CUDA mini-batch condensed training.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "actual_full_node_ratio", "condensed_nodes", "accuracy", "macro_f1", "training_time_s", "reason"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
