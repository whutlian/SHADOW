from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, markdown_table, write_csv
from shadow_hgc.preprop.block_budget import estimate_block_budget


FIELDS = [
    "dataset",
    "num_nodes",
    "num_edges",
    "num_classes",
    "block_set",
    "cache_mode",
    "total_cache_bytes",
    "feature_cache_bytes",
    "label_cache_bytes",
    "structure_cache_bytes",
    "full_edge_scans",
    "peak_cpu_ram_estimate_gb",
    "peak_gpu_ram_estimate_gb",
    "wall_time_category",
    "server_recommended",
    "uses_logits_as_input",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
    "ultra_policy",
]


def build_rows() -> list[dict]:
    specs = [
        ("ogbn-arxiv", 169343, 1166243, 40, 90941, 128),
        ("ogbn-products", 2449029, 123718280, 47, 196615, 128),
        ("ogbn-papers100M", 111059956, 1615685872, 172, 1207179, 64),
        ("MAG240M", 121751666, 17283641232, 153, 1112392, 64),
    ]
    blocks = ("X0", "X1", "X2", "X3", "Y1", "Y2", "Y3", "structure")
    rows: list[dict] = []
    for dataset, nodes, edges, classes, train_nodes, dim in specs:
        for row in estimate_block_budget(
            dataset=dataset,
            num_target_nodes=nodes,
            num_train_target_nodes=train_nodes,
            num_edges=edges,
            num_classes=classes,
            feature_dim=dim,
            selected_blocks=blocks,
        ):
            row["ultra_policy"] = "train_target_only_required" if dataset in {"ogbn-papers100M", "MAG240M"} else "local_feasible"
            if dataset in {"ogbn-papers100M", "MAG240M"} and row["cache_mode"] == "all_target_rows":
                row["server_recommended"] = True
                row["wall_time_category"] = "forbidden_by_t23_ultra_policy"
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="T23 ultra SFT dry-run.")
    parser.add_argument("--csv", default="experiments/tables/t23_scalability_dry_run_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t23_scalability_dry_run_summary.md")
    args = parser.parse_args()
    rows = build_rows()
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T23 Ultra SFT Dry-Run",
            "",
            *markdown_table(rows, ["dataset", "cache_mode", "total_cache_bytes", "full_edge_scans", "peak_cpu_ram_estimate_gb", "peak_gpu_ram_estimate_gb", "server_recommended", "ultra_policy"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": args.csv}, sort_keys=True))


if __name__ == "__main__":
    main()
