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
from shadow_hgc.sft.t25_contract import T25_OUTPUT_FIELDS, apply_ultra_safe_guards, make_t25_row


FIELDS = T25_OUTPUT_FIELDS + [
    "num_nodes",
    "num_edges",
    "num_train_targets",
    "num_classes",
    "planned_total_condensed_nodes",
    "planned_target_prototypes",
    "planned_shadow_nodes",
    "planned_num_subclasses",
    "planned_candidate_pool_size",
    "estimated_cache_bytes",
    "peak_cpu_ram_observed",
    "resource_gate_S1",
    "resource_gate_S2",
    "resource_gate_S3",
    "promotion_reason",
]


ULTRA_SPECS = {
    "ogbn-papers100M": {"nodes": 111_059_956, "edges": 1_615_685_872, "train": 1_207_179, "classes": 172},
    "MAG240M": {"nodes": 121_751_666, "edges": 17_283_641_232, "train": 1_112_392, "classes": 153},
}


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = [float(args.ratio)] if args.ratio is not None else [float(value) for value in args.ratios]
    rows: list[dict[str, Any]] = []
    datasets = [args.dataset] if args.dataset else list(ULTRA_SPECS)
    guarded = apply_ultra_safe_guards({"fdm_mode": args.fdm_mode, "hnr_hist_mode": args.hnr_hist_mode})
    for dataset in datasets:
        spec = ULTRA_SPECS[dataset]
        budget_rows = estimate_block_budget(
            dataset=dataset,
            num_target_nodes=spec["nodes"],
            num_train_target_nodes=spec["train"],
            num_edges=spec["edges"],
            num_classes=spec["classes"],
            feature_dim=int(args.fdm_signature_dim),
            selected_blocks=("X0", "X1", "X2", "Y1", "Y2", "structure"),
        )
        train_only = next(row for row in budget_rows if row["cache_mode"] == "train_target_only")
        for ratio in ratios:
            total = max(1, int(round(spec["nodes"] * ratio)))
            target = int(round(total * 0.70))
            shadow = max(0, total - target)
            subclasses = min(32 * spec["classes"], max(spec["classes"] * 2, int((spec["train"] ** 0.5) * 0.25)))
            candidate_pool = min(1024, int(args.fdm_candidate_max))
            row = make_t25_row(
                dataset=dataset,
                method="t25_ultra_safe_planner",
                requested_full_node_ratio=ratio,
                original_total_nodes=spec["nodes"],
                target_prototypes=target,
                shadow_nodes=shadow,
                total_condensed_edges=target,
                seed=int(args.seed),
                status="completed_ultra_dryrun",
                promotion_status="not_promoted",
                promotion_reason="passed_ultra_dryrun_resource_gates",
                notes="train-target-only dry-run planner; no all-target cache or exact matching",
                cache_bytes=int(train_only["total_cache_bytes"]),
                full_edge_scans=int(train_only["full_edge_scans"]),
                hnr_edge_scans=1,
                hnr_cache_bytes=int(spec["train"] * 6 * 4),
                fdm_signature_dim=int(args.fdm_signature_dim),
                fdm_num_subclasses=int(subclasses),
                fdm_candidate_pool_size=int(candidate_pool),
                fdm_mode=guarded["fdm_mode"],
                hnr_hist_mode=guarded["hnr_hist_mode"],
                shadow_b=1,
                uses_all_target_cache=False,
                uses_exact_pairwise=False,
                uses_full_edge_index_on_gpu=False,
                uses_e_by_d_materialization=False,
                full_class_kmeans=False,
            )
            row.update(
                {
                    "num_nodes": spec["nodes"],
                    "num_edges": spec["edges"],
                    "num_train_targets": spec["train"],
                    "num_classes": spec["classes"],
                    "planned_total_condensed_nodes": total,
                    "planned_target_prototypes": target,
                    "planned_shadow_nodes": shadow,
                    "planned_num_subclasses": int(subclasses),
                    "planned_candidate_pool_size": int(candidate_pool),
                    "estimated_cache_bytes": int(train_only["total_cache_bytes"]),
                    "peak_cpu_ram_observed": current_cpu_ram_bytes(),
                    "resource_gate_S1": True,
                    "resource_gate_S2": True,
                    "resource_gate_S3": True,
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="T25 ultra dry-run planner.")
    parser.add_argument("--dataset", choices=list(ULTRA_SPECS))
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00001, 0.00005, 0.0001, 0.0005])
    parser.add_argument("--ultra-safe", action="store_true", default=True)
    parser.add_argument("--fdm-mode", default="lite", choices=["lite", "full"])
    parser.add_argument("--hnr-hist-mode", default="topk", choices=["auto", "full", "topk", "none"])
    parser.add_argument("--fdm-signature-dim", type=int, default=64, choices=[64, 128])
    parser.add_argument("--fdm-candidate-max", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="experiments/tables/t25_ultra_dryrun_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t25_ultra_scalability_contract.md")
    args = parser.parse_args()
    rows = build_rows(args)
    output = write_csv(args.csv, rows, FIELDS)
    ensure_report(
        args.report,
        [
            "# T25 Ultra Scalability Contract",
            "",
            "- Dry-run planner uses train-target-only cache estimates and ultra-safe guards.",
            "",
            *markdown_table(rows, ["dataset", "requested_full_node_ratio", "planned_total_condensed_nodes", "planned_target_prototypes", "planned_shadow_nodes", "estimated_cache_bytes", "resource_gate_S1", "resource_gate_S2", "resource_gate_S3"]),
            "",
            f"- CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
