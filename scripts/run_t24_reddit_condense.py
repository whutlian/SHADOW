from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit import load_reddit_dataset
from shadow_hgc.ratio.scale_bucket import account_full_node_ratio


FIELDS = [
    "dataset",
    "method",
    "requested_full_node_ratio",
    "actual_full_node_ratio",
    "status",
    "reason",
    "accuracy",
    "macro_f1",
    "full_edge_scans",
    "preprop_cache_bytes",
    "condensed_nodes",
    "condensed_edges",
    "condensation_time_s",
    "training_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "uses_e_by_d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T24 Reddit SFT condensation table.")
    parser.add_argument("--reddit-root", default="dataset/Reddit")
    parser.add_argument("--csv", default="experiments/tables/t24_reddit_sft_condense_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t24_reddit_sft_condense_summary.md")
    args = parser.parse_args()
    try:
        graph = load_reddit_dataset(args.reddit_root)
        rows = []
        for ratio in [0.001, 0.0025, 0.005, 0.01]:
            total = max(1, int(round(graph.num_nodes * ratio)))
            target = max(graph.num_classes, int(round(total * 0.67)))
            shadow = max(0, total - target)
            accounting = account_full_node_ratio(original_total_nodes=graph.num_nodes, target_prototypes=target, shadow_nodes=shadow, condensed_edges=target * 2)
            for method in ["SFT-signature centroid", "SFT-signature herding", "SFT-signature medoid", "SFT-signature shadow condensed b=1", "b=2 ablation"]:
                rows.append(
                    {
                        "dataset": "Reddit",
                        "method": method,
                        "requested_full_node_ratio": ratio,
                        "status": "ready_not_trained",
                        "reason": "Reddit loader is complete; full X1-X3 preprop/condense run is resource-gated in this local stage",
                        "accuracy": "",
                        "macro_f1": "",
                        "full_edge_scans": 0,
                        "preprop_cache_bytes": int(graph.x.numel() * graph.x.element_size()),
                        "condensation_time_s": "",
                        "training_time_s": "",
                        "peak_cpu_ram_gb": "",
                        "peak_gpu_ram_gb": "",
                        "uses_e_by_d": False,
                        **accounting,
                    }
                )
    except Exception as exc:
        rows = [{"dataset": "Reddit", "method": "SFT-signature shadow condensed b=1", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "uses_e_by_d": False}]
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(args.report, ["# T24 Reddit SFT Condense", "", *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "actual_full_node_ratio", "condensed_nodes", "reason"]), "", f"- CSV: `{output}`"])
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
