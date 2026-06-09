from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run T0-S MAG240M resource estimate.")
    parser.add_argument("--dataset-root", default="D:/Shadow-HGC/dataset/ogbn_mag")
    parser.add_argument("--output", default="experiments/tables/t0s_mag240m_dry_run_seed42.json")
    parser.add_argument("--scap-topk", type=int, default=8)
    args = parser.parse_args()
    payload = {
        "dataset": "mag240m",
        "dataset_root": str(Path(args.dataset_root)),
        "status": "dry_run_estimate",
        "num_nodes": 121751666,
        "num_edges": 1728364232,
        "num_train_target_rows": 1112392,
        "feature_dim": 768,
        "num_classes": 153,
        "scap_topk": int(args.scap_topk),
        "full_edge_scans_per_relation": 2,
        "cache_all_targets": False,
        "uses_dense_e_by_d": False,
        "feature_policy": "train-period random projection and temporal-safe author initializer",
        "server_command": (
            "python scripts/run_t0s_fullgraph_parity.py --dataset mag240m "
            "--scap-topk 8 --streaming --temporal-safe-features --no-diffusion --no-dense-p2"
        ),
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
