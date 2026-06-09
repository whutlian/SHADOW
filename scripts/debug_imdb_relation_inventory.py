from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.data.small import load_processed_small_dataset_full_schema
from shadow_hgc.diagnostics.imdb_inventory import audit_imdb_relation_inventory, compare_imdb_clean_s1_to_sfb_metapaths


INVENTORY_FIELDS = [
    "dataset",
    "target_type",
    "all_node_types",
    "all_edge_types",
    "incoming_target_relations",
    "typed:directs_exists",
    "typed:acts_in_exists",
    "typed:keyword_in_exists",
    "MAM_available",
    "MDM_available",
    "MKM_available",
    "metapath_skipped",
    "feature_dims",
    "edge_counts_per_relation",
    "train_target_coverage_per_relation",
    "status",
]

EQUIV_FIELDS = [
    "dataset",
    "relation_name",
    "shape_a",
    "shape_b",
    "row_l2_mean",
    "row_l2_median",
    "row_l2_max",
    "cosine_mean",
    "cosine_min",
    "allclose_fraction",
    "block_dim",
    "block_norm_stats_source",
    "status",
]


def run_imdb_inventory_and_equivalence() -> tuple[dict, list[dict]]:
    graph = load_processed_small_dataset_full_schema("imdb")
    inventory = audit_imdb_relation_inventory(graph)
    required = [
        inventory["typed:directs_exists"],
        inventory["typed:acts_in_exists"],
        inventory["typed:keyword_in_exists"],
        inventory["MAM_available"],
        inventory["MDM_available"],
        inventory["MKM_available"],
    ]
    inventory["status"] = "completed" if all(required) else "blocked_by_relation_inventory_mismatch"
    metrics = compare_imdb_clean_s1_to_sfb_metapaths(graph, target_rows=torch.arange(graph.num_nodes[graph.target_type]))
    rows = []
    for name, row in metrics.items():
        passed = row["cosine_mean"] >= 0.999 and row["row_l2_mean"] <= 1e-4 and row["allclose_fraction"] >= 0.99
        row["status"] = "completed" if passed else "blocked_by_relation_inventory_mismatch"
        rows.append(row)
    return inventory, rows


def _write_report(inventory: dict, equiv_rows: list[dict], path: Path, inventory_csv: Path, equiv_csv: Path) -> None:
    lines = [
        "# IMDB Relation Inventory and Metapath Equivalence Seed 42",
        "",
        f"- Inventory status: `{inventory['status']}`",
        f"- typed:keyword_in exists: `{inventory['typed:keyword_in_exists']}`",
        f"- MAM/MDM/MKM available: `{inventory['MAM_available']}`, `{inventory['MDM_available']}`, `{inventory['MKM_available']}`",
        "",
        "| Block | Cosine Mean | Row L2 Mean | Allclose Fraction | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in equiv_rows:
        lines.append(
            f"| {row['relation_name']} | {row['cosine_mean']} | {row['row_l2_mean']} | {row['allclose_fraction']} | {row['status']} |"
        )
    lines.extend(["", f"- Inventory CSV: `{inventory_csv}`", f"- Equivalence CSV: `{equiv_csv}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IMDB relation inventory and clean S1 metapath equivalence.")
    parser.add_argument("--inventory-output", default="experiments/tables/imdb_relation_inventory_seed42.csv")
    parser.add_argument("--equivalence-output", default="experiments/tables/imdb_metapath_equivalence_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/imdb_relation_inventory_summary.md")
    args = parser.parse_args()
    inventory, equiv_rows = run_imdb_inventory_and_equivalence()
    inventory_output = Path(args.inventory_output)
    equiv_output = Path(args.equivalence_output)
    write_csv(inventory_output, [inventory], INVENTORY_FIELDS)
    write_csv(equiv_output, equiv_rows, EQUIV_FIELDS)
    _write_report(inventory, equiv_rows, Path(args.report), inventory_output, equiv_output)
    print(json.dumps({"inventory_status": inventory["status"], "equiv_rows": len(equiv_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
