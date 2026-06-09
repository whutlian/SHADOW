from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry_shape(entry: dict[str, Any], fallback: list[int]) -> list[int]:
    value = entry.get("shape", fallback)
    return [int(item) for item in value]


def inspect_memmap_manifest(memmap_root: str | Path) -> dict[str, Any]:
    root = Path(memmap_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {
            "dataset_present": False,
            "memmap_root": str(root),
            "reason": "manifest.json missing",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def entry(name: str) -> dict[str, Any]:
        return dict(manifest.get(name, {}))

    node_feat = entry("node_feat.npy")
    edge_index = entry("edge_index.npy")
    train_idx = entry("train_idx.npy")
    valid_idx = entry("valid_idx.npy")
    test_idx = entry("test_idx.npy")
    node_label = entry("node_label.npy")
    local_target_edges = entry("local_target_edges.npy")
    local_target_idx = entry("local_target_idx.npy")
    node_feat_shape = _entry_shape(node_feat, [0, 0])
    edge_shape = _entry_shape(edge_index, [0, 0])
    local_edge_shape = _entry_shape(local_target_edges, [0, 0])
    local_target_shape = _entry_shape(local_target_idx, [0])
    return {
        "dataset_present": True,
        "memmap_root": str(root),
        "manifest_path": str(manifest_path),
        "num_nodes": int(node_feat_shape[0]),
        "feature_dim": int(node_feat_shape[1]) if len(node_feat_shape) > 1 else 0,
        "num_edges": int(edge_shape[1]) if len(edge_shape) > 1 else 0,
        "train_nodes": int(_entry_shape(train_idx, [0])[0]),
        "valid_nodes": int(_entry_shape(valid_idx, [0])[0]),
        "test_nodes": int(_entry_shape(test_idx, [0])[0]),
        "node_feat_bytes": int(node_feat.get("bytes", 0)),
        "edge_index_bytes": int(edge_index.get("bytes", 0)),
        "label_bytes": int(node_label.get("bytes", 0)),
        "local_target_edges": int(local_edge_shape[1]) if len(local_edge_shape) > 1 else 0,
        "local_target_nodes": int(local_target_shape[0]),
        "manifest": manifest,
    }


def compute_trial_decision(
    *,
    peak_ram_estimate_bytes: int,
    available_ram_bytes: int,
    local_trial_status: str,
    guard_fraction: float = 0.8,
) -> dict[str, Any]:
    if local_trial_status not in {"completed", "completed_smoke"}:
        return {
            "full_scale_local_status": local_trial_status,
            "needs_server_run": True,
            "decision_reason": "local smoke trial did not complete",
        }
    guard_bytes = int(float(available_ram_bytes) * float(guard_fraction))
    if int(peak_ram_estimate_bytes) > guard_bytes:
        return {
            "full_scale_local_status": "blocked_resource_guard",
            "needs_server_run": True,
            "decision_reason": "estimated full-scale peak RAM exceeds local guard",
            "guard_fraction": float(guard_fraction),
            "guard_bytes": guard_bytes,
        }
    return {
        "full_scale_local_status": "local_feasible_by_estimate",
        "needs_server_run": False,
        "decision_reason": "estimated full-scale peak RAM is within local guard",
        "guard_fraction": float(guard_fraction),
        "guard_bytes": guard_bytes,
    }


def build_server_commands(*, dataset_root: str, output_dir: str) -> list[str]:
    py = r"C:\Users\slian\anaconda3\envs\pytorch\python.exe"
    return [
        (
            f"& '{py}' scripts/run_paper100m_local_trial.py "
            f"--dataset-root {dataset_root} --output-dir {output_dir} --seed 42 "
            "--sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion"
        ),
        (
            f"& '{py}' scripts/dry_run_ultra.py --dataset ogbn-papers100M "
            f"--ratios 0.001 0.0025 0.005 --output {output_dir}/paper100m_ultra_dry_run_server.json"
        ),
        (
            "python scripts/run_paper100m_local_trial.py "
            "--dataset-root /path/to/paper100M --output-dir experiments/logs/paper100m_server_seed42 "
            "--seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion"
        ),
    ]
