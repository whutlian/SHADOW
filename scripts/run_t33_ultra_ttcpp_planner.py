from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS
from shadow_hgc.sft.ultra_ttcpp_planner import plan_ultra_ttcpp


DATASETS = {
    "ogbn-papers100M": {"num_nodes": 111_059_956, "num_edges": 1_615_685_872, "num_classes": 172},
    "MAG240M": {"num_nodes": 121_751_666, "num_edges": 1_728_364_232, "num_classes": 153},
}


def build_ultra_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for ratio in args.ratios:
            for mode in args.teacher_cache_modes:
                for dim in args.signature_dims:
                    rows.append(
                        plan_ultra_ttcpp(
                            dataset=dataset,
                            requested_ratio=float(ratio),
                            teacher_cache_mode=str(mode),
                            signature_dim=int(dim),
                            **spec,
                        )
                    )
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_ultra_rows(args)
    csv_path = write_csv(args.csv, rows, T33_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T33 Ultra TTC++ Planner",
            "",
            *markdown_table(rows, ["dataset", "requested_full_node_ratio", "teacher_cache_mode", "planned_condensed_nodes", "teacher_topk_cache_bytes", "teacher_dense_cache_bytes_diagnostic", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 ultra-safe TTC++ top-k cache planner.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["ogbn-papers100M", "MAG240M"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00001, 0.00005, 0.0001, 0.0005])
    parser.add_argument("--teacher-cache-modes", nargs="+", default=["topk4_fp16", "topk8_fp16", "topk8_plus_entropy_margin"])
    parser.add_argument("--signature-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--ultra-safe", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_ultra_ttcpp_planner.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_ultra_ttcpp_planner.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
