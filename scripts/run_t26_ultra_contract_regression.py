from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes
from shadow_hgc.preprop.block_budget import estimate_block_budget
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row


ULTRA_SPECS = {
    "ogbn-papers100M": {"nodes": 111_059_956, "edges": 1_615_685_872, "train": 1_207_179, "classes": 172},
    "MAG240M": {"nodes": 121_751_666, "edges": 17_283_641_232, "train": 1_112_392, "classes": 153},
}

FIELDS = T26_REQUIRED_FIELDS + [
    "num_nodes",
    "num_edges",
    "num_train_targets",
    "num_classes",
    "planned_total_condensed_nodes",
    "planned_target_prototypes",
    "planned_shadow_nodes",
    "estimated_cache_bytes",
    "resource_gate_S1",
    "resource_gate_S2",
    "resource_gate_S3",
    "peak_cpu_ram_observed",
]


def build_rows(seed: int = 42, ratios: list[float] | None = None) -> list[dict[str, Any]]:
    ratios = [0.0001] if ratios is None else [float(value) for value in ratios]
    rows: list[dict[str, Any]] = []
    for dataset, spec in ULTRA_SPECS.items():
        budget_rows = estimate_block_budget(
            dataset=dataset,
            num_target_nodes=spec["nodes"],
            num_train_target_nodes=spec["train"],
            num_edges=spec["edges"],
            num_classes=spec["classes"],
            feature_dim=64,
            selected_blocks=("X0", "X1", "X2", "Y1", "Y2", "structure"),
        )
        train_only = next(row for row in budget_rows if row["cache_mode"] == "train_target_only")
        for ratio in ratios:
            total = max(1, int(round(spec["nodes"] * float(ratio))))
            target = int(round(total * 0.70))
            shadow = max(0, total - target)
            row = make_t26_row(
                dataset=dataset,
                method="t26_ultra_contract_regression",
                requested_full_node_ratio=ratio,
                original_total_nodes=spec["nodes"],
                target_prototypes=target,
                shadow_nodes=shadow,
                total_condensed_edges=target,
                seed=int(seed),
                status="completed_ultra_dryrun",
                promotion_status="not_promoted",
                promotion_reason="ultra_contract_regression_only",
                failure_reason="ultra_performance_not_run",
                notes="Train-target-only ultra dry-run; no all-target demand cache, exact pairwise, E x d materialization, or full edge_index GPU path.",
                cache_bytes=int(train_only["total_cache_bytes"]),
                disk_bytes=int(train_only["total_cache_bytes"]),
                full_edge_scans=int(train_only["full_edge_scans"]),
                edge_slice_cache_bytes=0,
                signature_dim=64,
                shadow_b=1,
                num_nodes=spec["nodes"],
                num_edges=spec["edges"],
                num_train_targets=spec["train"],
                num_classes=spec["classes"],
                planned_total_condensed_nodes=total,
                planned_target_prototypes=target,
                planned_shadow_nodes=shadow,
                estimated_cache_bytes=int(train_only["total_cache_bytes"]),
                resource_gate_S1=True,
                resource_gate_S2=True,
                resource_gate_S3=True,
                peak_cpu_ram_observed=current_cpu_ram_bytes(),
                uses_all_target_cache=False,
                uses_exact_pairwise=False,
                uses_e_by_d_materialization=False,
                uses_full_edge_index_on_gpu=False,
                full_class_kmeans=False,
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T26 ultra contract regression table.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--csv", default="experiments/tables/t26_ultra_contract_regression_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t26_ultra_contract_notes.md")
    args = parser.parse_args()
    rows = build_rows(seed=int(args.seed), ratios=[float(value) for value in args.ratios])
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T26 Ultra Contract Notes",
            "",
            "- Rows are dry-run contract regressions, not performance claims.",
            "- All forbidden ultra paths are false in generated rows.",
            "",
            *markdown_table(rows, ["dataset", "requested_full_node_ratio", "planned_total_condensed_nodes", "estimated_cache_bytes", "resource_gate_S1", "resource_gate_S2", "resource_gate_S3"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
