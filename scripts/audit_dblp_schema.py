from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.data.schema_audit import audit_dblp_schema
from shadow_hgc.data.small import load_processed_small_dataset


def _write_report(audit: dict, path: Path, csv_path: Path) -> None:
    lines = [
        "# DBLP Schema Audit Seed 42",
        "",
        f"- Target type: `{audit['target_type']}`",
        f"- Label node type: `{audit['label_node_type']}`",
        f"- APA available: `{audit['apa_available']}`",
        f"- Hard requirements passed: `{audit['hard_requirements_passed']}`",
        f"- Available target meta-path source types: `{', '.join(audit['available_source_types_for_target_metapaths'])}`",
        f"- Computed meta-path blocks: `{', '.join(audit['computed_metapath_blocks'])}`",
        f"- Skipped meta-path blocks: `{', '.join(audit['skipped_metapath_blocks'])}`",
        "",
        "## Edge Types",
        "",
        "| Source | Relation | Destination | Edges |",
        "|---|---|---|---:|",
    ]
    for edge in audit["edge_types"]:
        lines.append(f"| {edge['source_type']} | {edge['relation_name']} | {edge['destination_type']} | {edge['num_edges']} |")
    lines.extend([
        "",
        "## Label Distribution",
        "",
        f"- Train: `{json.dumps(audit['train_label_distribution'], sort_keys=True)}`",
        f"- Valid: `{json.dumps(audit['valid_label_distribution'], sort_keys=True)}`",
        f"- Test: `{json.dumps(audit['test_label_distribution'], sort_keys=True)}`",
        "",
        "## Note",
        "",
        audit["notes"],
        "",
        f"- CSV: `{csv_path}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit loaded DBLP target schema and meta-path availability.")
    parser.add_argument("--output", default="experiments/tables/dblp_schema_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/dblp_schema_audit_seed42.md")
    args = parser.parse_args()
    graph = load_processed_small_dataset("dblp")
    audit = audit_dblp_schema(graph)
    row = {
        "dataset": "dblp",
        "target_type": audit["target_type"],
        "label_node_type": audit["label_node_type"],
        "node_types": json.dumps(audit["node_types"]),
        "node_counts": json.dumps(audit["node_counts"], sort_keys=True),
        "edge_types": json.dumps(audit["edge_types"], sort_keys=True),
        "available_source_types_for_target_metapaths": json.dumps(audit["available_source_types_for_target_metapaths"]),
        "computed_metapath_blocks": json.dumps(audit["computed_metapath_blocks"]),
        "skipped_metapath_blocks": json.dumps(audit["skipped_metapath_blocks"]),
        "apa_available": audit["apa_available"],
        "train_label_distribution": json.dumps(audit["train_label_distribution"], sort_keys=True),
        "valid_label_distribution": json.dumps(audit["valid_label_distribution"], sort_keys=True),
        "test_label_distribution": json.dumps(audit["test_label_distribution"], sort_keys=True),
        "hard_requirements_passed": audit["hard_requirements_passed"],
        "notes": audit["notes"],
    }
    output = Path(args.output)
    write_csv(output, [row])
    _write_report(audit, Path(args.report), output)


if __name__ == "__main__":
    main()

