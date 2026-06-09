from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import estimate_block_cache_bytes, markdown_table, wall_time_category, write_csv, write_json


FIELDS = [
    "dataset",
    "target_type",
    "num_target_rows",
    "num_edges",
    "block_dim",
    "dtype_bytes",
    "x0_cache_bytes",
    "x1_cache_bytes",
    "x2_cache_bytes",
    "lad_scap_cache_bytes",
    "total_cache_bytes",
    "edge_scans",
    "full_edge_scans",
    "active_source_nodes_estimate",
    "cache_mode",
    "uses_memmap",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_logits_as_input",
    "uses_bounded_edges",
    "uses_diffusion_legacy",
    "wall_time_category",
    "server_recommended",
]


DATASET_ESTIMATES = {
    "ogbn-arxiv": {"target_type": "paper", "num_target_rows": 169343, "num_edges": 1166243, "num_classes": 40, "cache_mode": "all_target_rows"},
    "ogbn-products": {"target_type": "product", "num_target_rows": 2449029, "num_edges": 123718280, "num_classes": 47, "cache_mode": "all_target_rows"},
    "ogbn-papers100M": {"target_type": "paper", "num_target_rows": 111059956, "num_edges": 1615685872, "num_classes": 172, "cache_mode": "train_target_only"},
    "MAG240M": {"target_type": "paper", "num_target_rows": 121751666, "num_edges": 1728430000, "num_classes": 153, "cache_mode": "train_target_only"},
}


def dry_run_dataset(dataset: str, *, block_dim: int, dtype_bytes: int) -> dict[str, Any]:
    info = DATASET_ESTIMATES[dataset]
    rows = int(info["num_target_rows"])
    x0 = estimate_block_cache_bytes(rows, block_dim, dtype_bytes)
    x1 = estimate_block_cache_bytes(rows, block_dim, dtype_bytes) * 2
    x2 = estimate_block_cache_bytes(rows, block_dim, dtype_bytes) * 2
    lad = rows * int(info["num_classes"]) * dtype_bytes
    total = x0 + x1 + x2 + lad
    category = wall_time_category(int(info["num_edges"]))
    return {
        "dataset": dataset,
        "target_type": info["target_type"],
        "num_target_rows": rows,
        "num_edges": int(info["num_edges"]),
        "block_dim": int(block_dim),
        "dtype_bytes": int(dtype_bytes),
        "x0_cache_bytes": int(x0),
        "x1_cache_bytes": int(x1),
        "x2_cache_bytes": int(x2),
        "lad_scap_cache_bytes": int(lad),
        "total_cache_bytes": int(total),
        "edge_scans": 6,
        "full_edge_scans": 6,
        "active_source_nodes_estimate": rows,
        "cache_mode": info["cache_mode"],
        "uses_memmap": True,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_logits_as_input": False,
        "uses_bounded_edges": False,
        "uses_diffusion_legacy": False,
        "wall_time_category": category,
        "server_recommended": category == "server_recommended" or total > 24 * (1024**3),
    }


def _write_report(rows: list[dict[str, Any]], output: Path, report: Path) -> None:
    lines = [
        "# T2-SFT-NL Scalability Dry-Run",
        "",
        "Dry-runs estimate memmap cache size and scans only; they do not allocate dense P2, logits, bounded edges, or E x d tensors.",
        "",
        *markdown_table(rows, ["dataset", "cache_mode", "total_cache_bytes", "full_edge_scans", "wall_time_category", "server_recommended", "uses_logits_as_input", "uses_dense_p2"]),
        "",
        f"- CSV: `{output}`",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate T2 preprop/SFT scalability resources.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products", "ogbn-papers100M", "MAG240M"])
    parser.add_argument("--block-dim", type=int, default=64)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--output", default="experiments/tables/t2_sft_scalability_dry_run_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t2_sft_scalability_summary.md")
    args = parser.parse_args()
    rows = [dry_run_dataset(dataset, block_dim=args.block_dim, dtype_bytes=args.dtype_bytes) for dataset in args.datasets]
    output = write_csv(args.output, rows, FIELDS)
    write_json(Path(args.output).with_suffix(".json"), {"rows": rows})
    _write_report(rows, output, Path(args.report))
    print(json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
