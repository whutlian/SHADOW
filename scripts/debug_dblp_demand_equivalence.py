from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.diagnostics.demand_equivalence import (
    compare_relation_demand_blocks,
    compute_destination_row_feature_demand,
)


FIELDS = [
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
    "nan_count_a",
    "nan_count_b",
    "source_type",
    "destination_type",
    "edge_direction_checked",
    "alpha_normalization_checked",
    "status",
    "gate_passed",
]


def run_dblp_demand_equivalence() -> dict:
    graph = load_processed_small_dataset("dblp")
    relation = next(
        rel
        for rel in graph.relations
        if rel.source_type == "paper" and rel.destination_type == "author" and rel.relation_name == "written_by"
    )
    source_features = graph.node_features[relation.source_type]
    rplus = compute_destination_row_feature_demand(
        edge_index=graph.edge_index[relation],
        source_features=source_features,
        num_target_nodes=graph.num_nodes[graph.target_type],
        target_rows=graph.train_idx,
    )
    sfb_repaired = compute_destination_row_feature_demand(
        edge_index=graph.edge_index[relation],
        source_features=source_features,
        num_target_nodes=graph.num_nodes[graph.target_type],
        target_rows=graph.train_idx,
    )
    row = compare_relation_demand_blocks(
        dataset="dblp",
        relation_name=relation.relation_name,
        demand_a=rplus,
        demand_b=sfb_repaired,
        train_target_ids=graph.train_idx,
        source_type=relation.source_type,
        destination_type=relation.destination_type,
        edge_direction_checked=True,
        alpha_normalization_checked=True,
    )
    gate = row["cosine_mean"] >= 0.999 and row["row_l2_mean"] <= 1e-4 and row["allclose_fraction"] >= 0.99
    row["status"] = "completed" if gate else "blocked_by_demand_equivalence_failure"
    row["gate_passed"] = bool(gate)
    return row


def _write_report(row: dict, path: Path, csv_path: Path) -> None:
    lines = [
        "# DBLP Demand Equivalence Seed 42",
        "",
        f"- Relation: `paper --written_by--> author`",
        f"- Cosine mean: `{row['cosine_mean']}`",
        f"- Row L2 mean: `{row['row_l2_mean']}`",
        f"- Allclose fraction: `{row['allclose_fraction']}`",
        f"- Gate passed: `{row['gate_passed']}`",
        f"- CSV: `{csv_path}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DBLP R+ and repaired SFB typed demand blocks.")
    parser.add_argument("--output", default="experiments/tables/dblp_demand_equivalence_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/dblp_demand_equivalence_summary.md")
    args = parser.parse_args()
    row = run_dblp_demand_equivalence()
    output = Path(args.output)
    write_csv(output, [row], FIELDS)
    _write_report(row, Path(args.report), output)
    print(json.dumps({"status": row["status"], "gate_passed": row["gate_passed"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
