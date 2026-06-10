from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.preprop.block_budget import estimate_block_budget


FIELDS = ["dataset", "nodes", "edges", "target_train_nodes", "cache_mode", "selected_blocks", "cache_bytes", "edge_scans", "estimated_precompute_time", "server_recommended"]


def build_rows() -> list[dict]:
    specs = [("ogbn-papers100M", 111_059_956, 1_615_685_872, 1_207_179, 172), ("MAG240M", 121_751_666, 17_283_641_232, 1_112_392, 153)]
    rows: list[dict] = []
    blocks = ("X0", "X1", "X2", "Y1", "Y2", "structure")
    for dataset, nodes, edges, train_nodes, classes in specs:
        for row in estimate_block_budget(dataset=dataset, num_target_nodes=nodes, num_train_target_nodes=train_nodes, num_edges=edges, num_classes=classes, feature_dim=64, selected_blocks=blocks):
            rows.append(
                {
                    "dataset": dataset,
                    "nodes": nodes,
                    "edges": edges,
                    "target_train_nodes": train_nodes,
                    "cache_mode": row["cache_mode"],
                    "selected_blocks": row["block_set"],
                    "cache_bytes": row["total_cache_bytes"],
                    "edge_scans": row["full_edge_scans"],
                    "estimated_precompute_time": row["wall_time_category"],
                    "server_recommended": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="T24 ultra dry-run.")
    parser.add_argument("--csv", default="experiments/tables/t24_ultra_dry_run_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_ultra_dry_run_summary.md")
    args = parser.parse_args()
    rows = build_rows()
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(args.report, ["# T24 Ultra Dry-Run", "", *markdown_table(rows, FIELDS), "", f"- CSV: `{output}`"])
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
