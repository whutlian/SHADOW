from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.train.lazy_sft_memmap import train_lazy_sft_from_memmap


FIELDS = [
    "dataset",
    "target_type",
    "status",
    "reason",
    "run_mode",
    "model_type",
    "loss_type",
    "hidden_dim",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "prediction_entropy",
    "selected_blocks",
    "manifest_dir",
    "training_epochs",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "max_batch_materialized_bytes",
    "loads_edge_index",
    "uses_lazy_memmap",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "uses_diffusion_legacy",
]


def run_reddit_streaming_sft(args: argparse.Namespace) -> dict[str, Any]:
    labels, train_rows, valid_rows, test_rows = load_reddit_raw_memmap_labels_and_splits(args.memmap_root)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    selected_blocks = json.loads(args.selected_blocks)
    result = train_lazy_sft_from_memmap(
        manifest_dir=args.manifest_dir,
        labels=labels,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        num_classes=int(labels.max().item()) + 1,
        device=device,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        block_dropout=args.block_dropout,
        hop_dropout=args.hop_dropout,
        label_dropout=args.label_dropout,
        attention_heads=args.attention_heads,
        activation=args.activation,
        norm=args.norm,
        selected_blocks=selected_blocks,
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        two_stage=args.two_stage,
        stage1_loss=args.stage1_loss,
        stage2_loss=args.stage2_loss,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_lr_mult=args.stage2_lr_mult,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
    )
    summary = result.summary
    log_path = Path(args.log_dir) / f"reddit_streaming_sft_{args.model_type}_h{args.hidden_dim}_{args.loss_type}_e{summary.get('epochs_ran', args.epochs)}_seed{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    test = summary["test"]
    return {
        "dataset": "Reddit",
        "target_type": "reddit_node",
        "status": "completed_streaming_sft",
        "reason": "full streaming-preprop lazy memmap SFT training/eval completed",
        "run_mode": f"lazy_memmap_{device}",
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "hidden_dim": int(args.hidden_dim),
        "accuracy": test.get("accuracy", ""),
        "macro_f1": test.get("macro_f1", ""),
        "predicted_class_count": test.get("predicted_class_count", ""),
        "prediction_entropy": test.get("prediction_entropy", ""),
        "selected_blocks": json.dumps(list(summary.get("block_dims", {}).keys()), sort_keys=True),
        "manifest_dir": args.manifest_dir,
        "training_epochs": summary.get("epochs_ran", args.epochs),
        "training_time_s": summary.get("training_time_s", ""),
        "inference_time_s": summary.get("inference_time_s", ""),
        "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", ""),
        "max_batch_materialized_bytes": summary.get("max_batch_materialized_bytes", ""),
        "loads_edge_index": summary.get("loads_edge_index", False),
        "uses_lazy_memmap": summary.get("uses_lazy_memmap", True),
        "uses_logits_as_input": summary.get("uses_logits_as_input", False),
        "uses_teacher_logits": summary.get("uses_teacher_logits", False),
        "uses_kd": summary.get("uses_kd", False),
        "uses_dense_p2": summary.get("uses_dense_p2", False),
        "uses_bounded_edges": summary.get("uses_bounded_edges", False),
        "uses_e_by_d_materialization": summary.get("uses_e_by_d_materialization", False),
        "uses_diffusion_legacy": summary.get("uses_diffusion_legacy", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reddit lazy CPU/memmap + GPU mini-batch SFT from T24 streaming preprop.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-type", default="sagn_lite_v4")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Y1","Y2","Y3","structure"]')
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--hop-dropout", type=float, default=0.0)
    parser.add_argument("--label-dropout", type=float, default=0.05)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--norm", default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--stage1-loss", default="sqrt_weighted_ce")
    parser.add_argument("--stage2-loss", default="cross_entropy")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage2-lr-mult", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/tables/t24_reddit_streaming_sft_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_reddit_streaming_sft_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t24_reddit_streaming_sft_seed42")
    args = parser.parse_args()
    rows = [run_reddit_streaming_sft(args)]
    output = write_csv(args.output, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T24 Reddit Streaming SFT",
            "",
            "This row trains/evaluates on CPU/memmap-resident streaming preprop blocks with GPU mini-batches. It does not load the full edge index during training/eval.",
            "",
            *markdown_table(rows, ["dataset", "status", "run_mode", "model_type", "loss_type", "hidden_dim", "training_epochs", "accuracy", "macro_f1", "predicted_class_count", "peak_cpu_ram_gb", "peak_gpu_ram_gb"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "status": rows[0]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
