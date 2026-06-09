from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table, write_csv, write_json
from shadow_hgc.fullgraph.t0s_gates import required_scalability_fields, validate_scalability_resource_row


def _estimate_row(
    *,
    dataset: str,
    num_nodes: int,
    num_edges: int,
    num_train: int,
    num_classes: int,
    feature_dim: int,
    scap_topk: int,
    active_sources: int | None = None,
) -> dict:
    active = int(active_sources if active_sources is not None else min(num_nodes, max(num_train * 4, num_classes)))
    scap_cache_gb = (num_nodes * min(num_classes, scap_topk) * (4 + 2)) / (1024**3)
    demand_cache_gb = (num_train * 2 * feature_dim * 4) / (1024**3)
    disk_cache_gb = scap_cache_gb + demand_cache_gb
    row = {
        "dataset": dataset,
        "status": "dry_run_estimate",
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_target_rows": int(num_nodes),
        "num_train_target_rows": int(num_train),
        "num_active_sources": int(active),
        "num_classes": int(num_classes),
        "feature_dim": int(feature_dim),
        "scap_topk": int(scap_topk),
        "full_edge_scans": 2,
        "peak_cpu_ram_gb": float(min(64.0, 2.0 + scap_cache_gb + demand_cache_gb)),
        "peak_gpu_ram_gb": 0.0,
        "disk_cache_gb": float(disk_cache_gb),
        "scap_cache_gb": float(scap_cache_gb),
        "feature_demand_cache_gb": float(demand_cache_gb),
        "wall_time_s": 0.0,
        "edge_scan_throughput_edges_per_s": 0.0,
        "cache_all_targets": False,
        "uses_dense_e_by_d": False,
    }
    row.update(validate_scalability_resource_row(row))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Write T0-S scalability stress dry-run rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/tables/t0s_scalability_stress_seed42.csv")
    parser.add_argument("--json", default="experiments/tables/t0s_scalability_stress_seed42.json")
    parser.add_argument("--report", default="experiments/reports/t0s_scalability_stress_summary.md")
    args = parser.parse_args()
    rows = [
        _estimate_row(dataset="ogbn-arxiv", num_nodes=169343, num_edges=1166243 * 2, num_train=90941, num_classes=40, feature_dim=128, scap_topk=8),
        _estimate_row(dataset="ogbn-products", num_nodes=2449029, num_edges=61859140 * 2, num_train=196615, num_classes=47, feature_dim=100, scap_topk=8),
        _estimate_row(dataset="ogbn-papers100M", num_nodes=111059956, num_edges=1615685872, num_train=1207179, num_classes=172, feature_dim=128, scap_topk=8),
        _estimate_row(dataset="mag240m", num_nodes=121751666, num_edges=1728364232, num_train=1112392, num_classes=153, feature_dim=768, scap_topk=8),
    ]
    output = Path(args.output)
    write_csv(output, rows, fieldnames=["dataset", "status", *required_scalability_fields(), "valid", "reasons"])
    write_json(args.json, {"seed": args.seed, "rows": rows})
    lines = [
        "# T0-S Scalability Stress Seed 42",
        "",
        *markdown_table(rows, ["dataset", "status", "num_nodes", "num_edges", "scap_topk", "full_edge_scans", "disk_cache_gb", "valid", "reasons"]),
        "",
        f"- CSV: `{output}`",
        f"- JSON: `{args.json}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
