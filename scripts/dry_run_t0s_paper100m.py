from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import write_json
from shadow_hgc.ultra.paper100m_trial import inspect_memmap_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run T0-S paper100M resource estimate.")
    parser.add_argument("--dataset-root", default="D:/Shadow-HGC/dataset/paper100M")
    parser.add_argument("--output", default="experiments/tables/t0s_paper100m_dry_run_seed42.json")
    parser.add_argument("--scap-topk", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifest_root = root / "processed" / "papers100m_memmap"
    info = inspect_memmap_manifest(manifest_root if manifest_root.exists() else root)
    num_nodes = int(info.get("num_nodes") or 111059956)
    num_edges = int(info.get("num_edges") or 1615685872)
    train_nodes = int(info.get("train_nodes") or 1207179)
    feature_dim = int(info.get("feature_dim") or 128)
    payload = {
        "dataset": "ogbn-papers100M",
        "dataset_root": str(root),
        "dataset_present": bool(info.get("dataset_present", False)),
        "status": "dry_run_estimate",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_train_target_rows": train_nodes,
        "feature_dim": feature_dim,
        "num_classes": 172,
        "scap_topk": int(args.scap_topk),
        "full_edge_scans": 2,
        "cache_all_targets": False,
        "uses_dense_e_by_d": False,
        "server_command": (
            "python scripts/run_t0s_fullgraph_parity.py --dataset ogbn-papers100M "
            "--scap-topk 8 --streaming --no-diffusion --no-dense-p2"
        ),
        "manifest": info,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
