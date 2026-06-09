from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv


FIELDS = ["dataset", "num_target_nodes", "num_classes", "all_target_logit_cache_gb", "topk_pseudo_scap_cache_gb", "edge_scans", "memmap_required", "topk_sparse_required", "wall_time_category"]


def run(args) -> list[dict]:
    specs = [
        ("ogbn-arxiv", 169343, 40, "local_feasible"),
        ("ogbn-products", 2449029, 47, "local_feasible"),
        ("ogbn-papers100M", 111059956, 172, "server_recommended"),
        ("MAG240M", 121751666, 153, "server_recommended"),
    ]
    rows = []
    for dataset, n, c, wall in specs:
        logit_bytes = int(n * c * 2)
        k = min(8, c)
        topk_bytes = int(n * k * (2 + 8) + n * 12)
        rows.append(
            {
                "dataset": dataset,
                "num_target_nodes": n,
                "num_classes": c,
                "all_target_logit_cache_gb": round(logit_bytes / 1e9, 4),
                "topk_pseudo_scap_cache_gb": round(topk_bytes / 1e9, 4),
                "edge_scans": 2,
                "memmap_required": True,
                "topk_sparse_required": True,
                "wall_time_category": wall,
            }
        )
    write_csv(args.output, rows, FIELDS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="T1.1 large logit-affinity dry-run.")
    parser.add_argument("--output", default="experiments/tables/t1_large_logit_affinity_dry_run_seed42.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
