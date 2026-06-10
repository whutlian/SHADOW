from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.train.lazy_sft_memmap import load_arxiv_labels_and_splits, train_lazy_sft_from_memmap


DEFAULT_BLOCKS = [
    "X0",
    "X1_cite_ref",
    "X1_cited_by",
    "X2_cite_ref",
    "X2_cited_by",
    "X3_mix",
    "Xres1_cite_ref",
    "Xres1_cited_by",
    "structure",
    "Y1_cite_ref",
    "Y1_cited_by",
    "Y2_cite_ref",
    "Y2_cited_by",
    "Y3_mix",
]

FIELDS = [
    "dataset",
    "variant",
    "status",
    "seed",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "valid_acc",
    "valid_macro_f1",
    "teacher_gate_A1",
    "selected_blocks",
    "train_label_scope",
    "manifest_dir",
    "model_type",
    "hidden_dim",
    "epochs",
    "two_stage",
    "loss_type",
    "dropout",
    "block_dropout",
    "hop_dropout",
    "label_dropout",
    "lr",
    "weight_decay",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "cache_bytes",
    "full_edge_scans",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "uses_diffusion_legacy",
    "notes",
]


def _row_from_summary(
    *,
    variant: str,
    summary: dict[str, Any],
    selected_blocks: list[str],
    manifest_dir: str | Path,
    seed: int,
    hidden_dim: int | str = "",
    dropout: float | str = "",
    block_dropout: float | str = "",
    hop_dropout: float | str = "",
    label_dropout: float | str = "",
    lr: float | str = "",
    weight_decay: float | str = "",
    train_label_scope: str = "train_only",
) -> dict[str, Any]:
    test = summary.get("test", {})
    valid = summary.get("valid", {})
    acc = float(test.get("accuracy", 0.0) or 0.0)
    return {
        "dataset": "ogbn-arxiv",
        "variant": str(variant),
        "status": "completed_long",
        "seed": int(seed),
        "accuracy": acc,
        "macro_f1": test.get("macro_f1", ""),
        "predicted_class_count": test.get("predicted_class_count", ""),
        "valid_acc": valid.get("accuracy", ""),
        "valid_macro_f1": valid.get("macro_f1", ""),
        "teacher_gate_A1": bool(acc >= 0.715),
        "selected_blocks": json.dumps(list(selected_blocks), sort_keys=True),
        "train_label_scope": str(train_label_scope),
        "manifest_dir": str(manifest_dir),
        "model_type": summary.get("model_type", ""),
        "hidden_dim": hidden_dim,
        "epochs": int(summary.get("epochs_ran", 0) or 0),
        "two_stage": bool(summary.get("two_stage", False)),
        "loss_type": summary.get("loss_type", ""),
        "dropout": dropout,
        "block_dropout": block_dropout,
        "hop_dropout": hop_dropout,
        "label_dropout": label_dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "training_time_s": summary.get("training_time_s", ""),
        "inference_time_s": summary.get("inference_time_s", ""),
        "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", ""),
        "cache_bytes": summary.get("max_batch_materialized_bytes", ""),
        "full_edge_scans": 0,
        "uses_logits_as_input": bool(summary.get("uses_logits_as_input", False)),
        "uses_teacher_logits": bool(summary.get("uses_teacher_logits", False)),
        "uses_kd": bool(summary.get("uses_kd", False)),
        "uses_dense_p2": bool(summary.get("uses_dense_p2", False)),
        "uses_bounded_edges": bool(summary.get("uses_bounded_edges", False)),
        "uses_e_by_d_materialization": bool(summary.get("uses_e_by_d_materialization", False)),
        "uses_diffusion_legacy": bool(summary.get("uses_diffusion_legacy", False)),
        "notes": f"real lazy-memmap arxiv teacher run; train_label_scope={train_label_scope}; no logits/KD/dense-P2/E-by-d inputs",
    }


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    labels, train_rows, valid_rows, test_rows = load_arxiv_labels_and_splits(args.dataset_root)
    try:
        selected_blocks = json.loads(args.selected_blocks)
    except json.JSONDecodeError:
        selected_blocks = [item.strip().strip('"').strip("'") for item in str(args.selected_blocks).strip("[]").split(",") if item.strip()]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    fit_train_rows = torch.unique(torch.cat([train_rows, valid_rows]).to(torch.long), sorted=True) if bool(args.train_plus_valid) else train_rows
    result = train_lazy_sft_from_memmap(
        manifest_dir=args.manifest_dir,
        labels=labels,
        train_rows=fit_train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        num_classes=int(labels.max().item()) + 1,
        device=device,
        model_type=args.model_type,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        num_layers=int(args.num_layers),
        block_dropout=float(args.block_dropout),
        hop_dropout=float(args.hop_dropout),
        label_dropout=float(args.label_dropout),
        selected_blocks=selected_blocks,
        loss_type=args.loss_type,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        seed=int(args.seed),
        label_smoothing=float(args.label_smoothing),
    )
    row = _row_from_summary(
        variant=args.variant,
        summary=result.summary,
        selected_blocks=selected_blocks,
        manifest_dir=args.manifest_dir,
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        block_dropout=float(args.block_dropout),
        hop_dropout=float(args.hop_dropout),
        label_dropout=float(args.label_dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        train_label_scope="train_plus_valid" if bool(args.train_plus_valid) else "train_only",
    )
    log_path = Path(args.log_dir) / f"{args.variant}_seed{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = [existing for existing in read_csv(args.csv) if existing.get("variant") != args.variant] if args.append else []
    rows.append(row)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Arxiv Actual Teacher Runs",
            "",
            "- Rows are real lazy-memmap SFT teacher training runs.",
            "- A1 gate is `accuracy >= 0.715`.",
            "",
            *markdown_table(rows, ["variant", "status", "accuracy", "macro_f1", "predicted_class_count", "valid_acc", "teacher_gate_A1", "model_type", "hidden_dim", "epochs"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    return {"status": "completed", "row": row, "csv": str(output), "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real T26 ogbn-arxiv teacher candidate.")
    parser.add_argument("--variant", default="A1_real_sagn_lite_v4_h768")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_arxiv_seed42")
    parser.add_argument("--selected-blocks", default=json.dumps(DEFAULT_BLOCKS))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-type", default="sagn_lite_v4")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--hop-dropout", type=float, default=0.0)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--loss-type", default="cross_entropy")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-plus-valid", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_arxiv_teacher_actual.md")
    parser.add_argument("--log-dir", default="experiments/logs/t26_arxiv_teacher_actual_seed42")
    args = parser.parse_args()
    print(json.dumps(run_actual(args), sort_keys=True))


if __name__ == "__main__":
    main()
