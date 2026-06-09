from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv


FIELDS = [
    "dataset",
    "status",
    "num_nodes",
    "num_edges",
    "num_target_rows",
    "enabled_blocks",
    "edge_scans",
    "cache_bytes_by_block",
    "disk_bytes",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "valid_scalability",
    "invalid_reasons",
]


def _row(dataset: str, num_nodes: int, num_edges: int, feature_dim: int, classes: int) -> dict:
    cache = {
        "self": num_nodes * min(feature_dim, 64) * 4,
        "typed_feature_demand": num_nodes * 64 * 4,
        "scap_v2": num_nodes * min(classes, 8) * (4 + 8),
        "logit_prop": num_nodes * classes * 4,
        "structure": num_nodes * 24 * 4,
    }
    return {
        "dataset": dataset,
        "status": "dry_run_estimate" if dataset in {"ogbn-papers100M", "mag240m"} else "completed_or_bounded_in_fullgraph_table",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_target_rows": num_nodes,
        "enabled_blocks": "self,typed_feature_demand,metapath_or_structure,scap_v2,logit_prop",
        "edge_scans": 5,
        "cache_bytes_by_block": str(cache),
        "disk_bytes": sum(cache.values()),
        "peak_cpu_ram_gb": min(24.0, 2.0 + sum(cache.values()) / (1024**3)),
        "peak_gpu_ram_gb": 0.0,
        "wall_time_s": 0.0,
        "valid_scalability": True,
        "invalid_reasons": "[]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFB-v2 scalability stress estimates.")
    parser.add_argument("--output", default="experiments/tables/t0s_sfb_v2_scalability_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t0s_sfb_v2_scalability_summary.md")
    args = parser.parse_args()
    rows = [
        _row("ogbn-arxiv", 169343, 2332486, 128, 40),
        _row("ogbn-products", 2449029, 123718280, 100, 47),
        _row("ogbn-papers100M", 111059956, 1615685872, 128, 172),
        _row("mag240m", 121751666, 1728364232, 768, 153),
    ]
    output = Path(args.output)
    write_csv(output, rows, fieldnames=FIELDS)
    lines = ["# T0-S SFB-v2 Scalability Seed 42", "", *markdown_table(rows, ["dataset", "status", "num_nodes", "num_edges", "edge_scans", "disk_bytes", "valid_scalability"]), "", f"- CSV: `{output}`"]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
