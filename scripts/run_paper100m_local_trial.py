from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.demand.cache import estimate_ultra_dry_run
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.paper100m_trial import build_server_commands, compute_trial_decision, inspect_memmap_manifest


def _available_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _sample_ids(ids: np.ndarray, *, sample_size: int, seed: int) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.shape[0] <= sample_size:
        return ids.astype(np.int64, copy=False)
    rng = np.random.default_rng(seed)
    pos = rng.choice(ids.shape[0], size=sample_size, replace=False)
    pos.sort()
    return ids[pos].astype(np.int64, copy=False)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).to(torch.float32).mean().item()) if labels.numel() else 0.0


def _train_linear_probe(
    *,
    memmap_root: Path,
    sample_train: int,
    sample_valid: int,
    seed: int,
    epochs: int,
    lr: float,
    device_name: str,
) -> dict:
    node_feat = np.load(memmap_root / "node_feat.npy", mmap_mode="r")
    node_label = np.load(memmap_root / "node_label.npy", mmap_mode="r")
    train_idx = np.load(memmap_root / "train_idx.npy", mmap_mode="r")
    valid_idx = np.load(memmap_root / "valid_idx.npy", mmap_mode="r")
    train_nodes = _sample_ids(train_idx, sample_size=sample_train, seed=seed)
    valid_nodes = _sample_ids(valid_idx, sample_size=sample_valid, seed=seed + 1)
    x_train = np.asarray(node_feat[train_nodes], dtype=np.float32)
    y_train = np.asarray(node_label[train_nodes]).reshape(-1).astype(np.int64)
    x_valid = np.asarray(node_feat[valid_nodes], dtype=np.float32)
    y_valid = np.asarray(node_label[valid_nodes]).reshape(-1).astype(np.int64)
    valid_label_mask = (y_train >= 0) & (y_train < 10000)
    x_train = x_train[valid_label_mask]
    y_train = y_train[valid_label_mask]
    valid_eval_mask = (y_valid >= 0) & (y_valid < 10000)
    x_valid = x_valid[valid_eval_mask]
    y_valid = y_valid[valid_eval_mask]
    num_classes = int(max(y_train.max(initial=0), y_valid.max(initial=0)) + 1)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    valid_x = torch.from_numpy(x_valid).to(device)
    valid_y = torch.from_numpy(y_valid).to(device)
    start = time.perf_counter()
    first_loss = None
    final_loss = None
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        logits = model(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        if first_loss is None:
            first_loss = float(loss.detach().cpu().item())
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    train_time = time.perf_counter() - start
    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        valid_logits = model(valid_x)
    infer_time = time.perf_counter() - infer_start
    return {
        "status": "completed_smoke",
        "sample_train": int(train_x.shape[0]),
        "sample_valid": int(valid_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
        "num_classes_observed": int(num_classes),
        "train_accuracy": _accuracy(train_logits, train_y),
        "valid_accuracy": _accuracy(valid_logits, valid_y),
        "train_loss_start": first_loss,
        "train_loss_end": final_loss,
        "training_time_s": float(train_time),
        "inference_time_s": float(infer_time),
        "device": str(device),
    }


def _estimate_full_scale(manifest_info: dict, *, dtype_bytes: int = 4) -> dict:
    num_train = int(manifest_info.get("train_nodes", 0))
    num_nodes = int(manifest_info.get("num_nodes", 0))
    feature_dim = int(manifest_info.get("feature_dim", 128))
    num_edges = int(manifest_info.get("num_edges", 0))
    train_train_edges = int(manifest_info.get("local_target_edges", 0))
    relations = [
        {
            "name": "paper_cite_ref_paper",
            "num_edges": num_edges,
            "num_train_target_incident_edges": train_train_edges,
            "num_train_train_edges": train_train_edges,
            "num_active_sources": num_nodes,
            "num_source_nodes": num_nodes,
            "num_target_nodes": num_nodes,
            "source_is_target": True,
        },
        {
            "name": "paper_cited_by_paper",
            "num_edges": num_edges,
            "num_train_target_incident_edges": train_train_edges,
            "num_train_train_edges": train_train_edges,
            "num_active_sources": num_nodes,
            "num_source_nodes": num_nodes,
            "num_target_nodes": num_nodes,
            "source_is_target": True,
        },
    ]
    return estimate_ultra_dry_run(
        num_train_targets=num_train,
        feature_dim=feature_dim,
        dtype_bytes=dtype_bytes,
        relations=relations,
        dense_map_budget_bytes=64 * 1024 * 1024,
        num_target_nodes=num_nodes,
        num_source_nodes=num_nodes,
    )


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# paper100M Local Trial Seed 42",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in [
        "status",
        "dataset_root",
        "memmap_root",
        "sample_train",
        "sample_valid",
        "train_accuracy",
        "valid_accuracy",
        "full_scale_local_status",
        "needs_server_run",
        "peak_ram_estimate_gb",
        "available_ram_gb",
        "full_edge_scans",
        "disk_spill_used",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Server Commands", ""])
    for command in summary.get("server_commands", []):
        lines.append(f"```powershell\n{command}\n```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded local ogbn-papers100M memmap smoke trial.")
    parser.add_argument("--dataset-root", default="D:/Shadow-HGC/dataset/paper100M")
    parser.add_argument("--output-dir", default="experiments/logs/paper100m_local_trial_seed42")
    parser.add_argument("--table", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-train", type=int, default=20_000)
    parser.add_argument("--sample-valid", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-scale", action="store_true")
    parser.add_argument("--no-diffusion", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    memmap_root = dataset_root / "processed" / "papers100m_memmap"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paper100m_local_trial_seed42.json"
    csv_path = Path(args.table) if args.table else output_dir / "paper100m_local_trial_seed42.csv"
    report_path = Path(args.report) if args.report else output_dir / "paper100m_local_trial_seed42.md"

    manifest_info = inspect_memmap_manifest(memmap_root)
    available_ram = _available_ram_bytes()
    full_estimate = _estimate_full_scale(manifest_info) if manifest_info.get("dataset_present") else {}
    try:
        local = _train_linear_probe(
            memmap_root=memmap_root,
            sample_train=args.sample_train,
            sample_valid=args.sample_valid,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            device_name=args.device,
        ) if manifest_info.get("dataset_present") else {"status": "dataset_missing"}
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() else "runtime_error"
        local = {"status": status, "reason": message}
    except MemoryError as exc:
        local = {"status": "oom", "reason": str(exc)}

    peak_ram = int(full_estimate.get("peak_ram_estimate_bytes", 0))
    decision = compute_trial_decision(
        peak_ram_estimate_bytes=peak_ram,
        available_ram_bytes=available_ram,
        local_trial_status=str(local.get("status", "")),
    )
    server_commands = build_server_commands(dataset_root=str(dataset_root).replace("\\", "/"), output_dir=str(output_dir).replace("\\", "/"))
    summary = {
        "dataset": "ogbn-papers100M",
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "memmap_root": str(memmap_root),
        "no_diffusion": bool(args.no_diffusion),
        "full_scale_requested": bool(args.full_scale),
        "status": local.get("status", ""),
        **manifest_info,
        **local,
        **decision,
        "peak_ram_estimate_bytes": peak_ram,
        "peak_ram_estimate_gb": peak_ram / 1e9 if peak_ram else 0.0,
        "available_ram_bytes": available_ram,
        "available_ram_gb": available_ram / 1e9 if available_ram else 0.0,
        "full_edge_scans": full_estimate.get("total_expected_full_edge_scans", ""),
        "edge_slice_cache_bytes": full_estimate.get("edge_slice_cache_bytes", ""),
        "disk_spill_estimate_bytes": full_estimate.get("disk_spill_estimate_bytes", ""),
        "disk_spill_used": full_estimate.get("disk_spill_used", ""),
        "peak_cpu_ram_mb": current_cpu_ram_bytes() / (1024**2),
        "peak_gpu_ram_mb": current_gpu_ram_bytes() / (1024**2),
        "server_commands": server_commands if decision.get("needs_server_run") else [],
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
        if key not in {"manifest"}
    }
    _write_csv(csv_path, row)
    _write_report(report_path, summary)
    print(f"wrote {json_path}")
    print(f"status={summary['status']}")
    print(f"full_scale_local_status={summary['full_scale_local_status']}")
    if summary.get("needs_server_run"):
        print("server_run_required=true")


if __name__ == "__main__":
    main()
