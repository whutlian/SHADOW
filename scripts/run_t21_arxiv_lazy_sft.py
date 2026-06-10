from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t21_common import markdown_table, write_csv
from shadow_hgc.train.lazy_sft_memmap import load_arxiv_labels_and_splits, train_lazy_sft_from_memmap


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
    "selected_blocks",
    "manifest_dir",
    "training_epochs",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "uses_diffusion_legacy",
]


def run_arxiv(args) -> dict[str, Any]:
    labels, train_rows, valid_rows, test_rows = load_arxiv_labels_and_splits(args.dataset_root)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
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
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
    )
    summary = result.summary
    log_path = Path(args.log_dir) / f"ogbn_arxiv_lazy_sft_{args.model_type}_h{args.hidden_dim}_{args.loss_type}_e{args.epochs}_seed{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    test = summary["test"]
    return {
        "dataset": "ogbn-arxiv",
        "target_type": "paper",
        "status": "completed",
        "reason": "lazy_memmap_gpu_sft_completed" if str(device).startswith("cuda") else "lazy_memmap_cpu_sft_completed",
        "run_mode": f"lazy_memmap_{device}",
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "hidden_dim": args.hidden_dim,
        "accuracy": test.get("accuracy", ""),
        "macro_f1": test.get("macro_f1", ""),
        "predicted_class_count": test.get("predicted_class_count", ""),
        "selected_blocks": json.dumps(list(summary.get("block_dims", {}).keys()), sort_keys=True),
        "manifest_dir": args.manifest_dir,
        "training_epochs": args.epochs,
        "training_time_s": summary.get("training_time_s", ""),
        "inference_time_s": summary.get("inference_time_s", ""),
        "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", ""),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "uses_diffusion_legacy": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ogbn-arxiv lazy CPU/memmap + GPU mini-batch SFT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", default="experiments/preprop/t21_seed42/ogbn-arxiv")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-type", default="gamlp_lite", choices=["sagn_lite", "gamlp_lite", "residual_block_gated"])
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/tables/t21_arxiv_lazy_sft_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_arxiv_lazy_sft_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t21_arxiv_lazy_sft_seed42")
    args = parser.parse_args()
    rows = [run_arxiv(args)]
    output = write_csv(args.output, rows, FIELDS)
    lines = [
        "# T2.1 ogbn-arxiv Lazy SFT",
        "",
        "This row uses CPU/memmap-resident T2.1 preprop blocks and GPU mini-batch SFT. It does not load full edge_index during training/eval.",
        "",
        *markdown_table(rows, ["dataset", "status", "run_mode", "model_type", "loss_type", "hidden_dim", "accuracy", "macro_f1", "predicted_class_count", "peak_cpu_ram_gb", "peak_gpu_ram_gb"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "status": rows[0]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
