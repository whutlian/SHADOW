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
from shadow_hgc.data.reddit import load_reddit_dataset
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.train.train_sft_teacher import sft_loss
from shadow_hgc.eval.sft_eval import sft_metrics


FIELDS = [
    "dataset",
    "model",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "precompute_time_s",
    "train_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "cache_bytes",
    "full_edge_scans",
    "num_nodes",
    "num_edges",
    "feature_dim",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "uses_e_by_d",
]


def _train_x0(graph, *, epochs: int, hidden_dim: int, device: str) -> tuple[dict[str, Any], float]:
    target_device = torch.device(device)
    blocks = {"self": graph.x}
    model = SFTTeacherV3({"self": graph.feature_dim}, num_classes=graph.num_classes, model_type="sagn_lite_v4", hidden_dim=hidden_dim, dropout=0.3).to(target_device)
    model.fit_block_stats(blocks, train_rows=graph.train_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    started = time.perf_counter()
    labels = graph.y
    for epoch in range(int(epochs)):
        order = graph.train_idx[torch.randperm(graph.train_idx.numel(), generator=torch.Generator().manual_seed(42 + epoch))]
        for start in range(0, int(order.numel()), 16384):
            rows = order[start : start + 16384]
            opt.zero_grad(set_to_none=True)
            logits = model({"self": graph.x[rows].to(target_device)})
            loss = sft_loss(logits, labels[rows].to(target_device), loss_type="cross_entropy", train_labels=labels[graph.train_idx].to(target_device))
            loss.backward()
            opt.step()
    train_time = float(time.perf_counter() - started)
    model.eval()
    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(graph.test_idx.numel()), 65536):
            rows = graph.test_idx[start : start + 65536]
            logits_list.append(model({"self": graph.x[rows].to(target_device)}).detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    metrics = sft_metrics(logits, graph.y[graph.test_idx], torch.arange(graph.test_idx.numel()), num_classes=graph.num_classes)
    return metrics, train_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 Reddit SFT fullgraph onboarding.")
    parser.add_argument("--reddit-root", default="dataset/Reddit")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default="experiments/tables/t24_reddit_sft_fullgraph_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_reddit_sft_fullgraph_summary.md")
    args = parser.parse_args()
    try:
        graph = load_reddit_dataset(args.reddit_root)
        metrics: dict[str, Any] = {}
        train_time: float | str = ""
        status = "loaded_ready"
        reason = "processed Reddit cache loaded; use --train to run X0 table SFT smoke"
        if args.train:
            metrics, train_time = _train_x0(graph, epochs=args.epochs, hidden_dim=args.hidden_dim, device=args.device)
            status = "completed_x0_sft_smoke"
            reason = "X0-only table SFT smoke completed; full X1-X3 preprop is not run by default"
        row = {
            "dataset": "Reddit",
            "model": "sagn_lite_v4_x0_smoke",
            "status": status,
            "reason": reason,
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "predicted_class_count": metrics.get("predicted_class_count", ""),
            "precompute_time_s": 0,
            "train_time_s": train_time,
            "peak_cpu_ram_gb": "",
            "peak_gpu_ram_gb": "",
            "cache_bytes": int(graph.x.numel() * graph.x.element_size()),
            "full_edge_scans": 0,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "feature_dim": graph.feature_dim,
            "train_nodes": int(graph.train_idx.numel()),
            "valid_nodes": int(graph.valid_idx.numel()),
            "test_nodes": int(graph.test_idx.numel()),
            "uses_e_by_d": False,
        }
    except Exception as exc:
        row = {"dataset": "Reddit", "model": "sagn_lite_v4", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "uses_e_by_d": False}
    rows = [row]
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(args.report, ["# T24 Reddit SFT Fullgraph", "", *markdown_table(rows, ["dataset", "model", "status", "accuracy", "macro_f1", "num_nodes", "num_edges", "reason"]), "", f"- CSV: `{output}`"])
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
