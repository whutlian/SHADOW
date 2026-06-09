from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.schema_audit import audit_schema_alignment
from shadow_hgc.data.small import load_processed_small_dataset, load_processed_small_dataset_full_schema


FIELDS = [
    "dataset",
    "source",
    "loader_name",
    "target_type",
    "label_node_type",
    "node_types",
    "edge_types",
    "num_nodes_by_type",
    "num_edges_by_type",
    "metapath_available",
    "metapath_missing",
    "missing_reason",
    "split_hash",
    "feature_hash",
    "label_hash",
    "schema_hash",
    "freehgc_or_hgb_alignment_status",
    "notes",
]


def _jsonable_row(row: dict) -> dict:
    out = dict(row)
    for key in ("node_types", "edge_types", "num_nodes_by_type", "num_edges_by_type", "metapath_available", "metapath_missing"):
        out[key] = json.dumps(out.get(key, {}), sort_keys=True)
    return out


def _medium_row(dataset: str, *, source: str, reason: str) -> dict:
    return {
        "dataset": dataset,
        "source": source,
        "loader_name": "ogb_homogeneous",
        "target_type": "paper" if dataset == "ogbn-arxiv" else "product",
        "label_node_type": "paper" if dataset == "ogbn-arxiv" else "product",
        "node_types": json.dumps(["paper" if dataset == "ogbn-arxiv" else "product"]),
        "edge_types": json.dumps([]),
        "num_nodes_by_type": json.dumps({}),
        "num_edges_by_type": json.dumps({}),
        "metapath_available": json.dumps([]),
        "metapath_missing": json.dumps([]),
        "missing_reason": "",
        "split_hash": "homogeneous_not_loaded" if reason else "homogeneous",
        "feature_hash": "homogeneous_not_loaded" if reason else "homogeneous",
        "label_hash": "homogeneous_not_loaded" if reason else "homogeneous",
        "schema_hash": "homogeneous",
        "freehgc_or_hgb_alignment_status": "not_applicable",
        "notes": reason or "homogeneous/near-homogeneous OGB dataset; hetero meta-path schema is not applicable",
    }


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# Schema Alignment Audit Seed 42",
        "",
        "| Dataset | Loader | Target | Available | Missing | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['loader_name']} | {row['target_type']} | "
            f"{row['metapath_available']} | {row['metapath_missing']} | "
            f"{row['freehgc_or_hgb_alignment_status']} | {row['notes']} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run schema/data alignment audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="experiments/tables/schema_alignment_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/schema_alignment_audit_seed42.md")
    args = parser.parse_args()
    rows: list[dict] = []
    for dataset in ["acm", "dblp", "imdb"]:
        rows.append(_jsonable_row(audit_schema_alignment(load_processed_small_dataset(dataset), loader_name="current_processed", source="local_processed")))
        rows.append(_jsonable_row(audit_schema_alignment(load_processed_small_dataset_full_schema(dataset), loader_name="full_schema", source="local_processed")))
    for dataset in ["ogbn-arxiv", "ogbn-products"]:
        rows.append(_medium_row(dataset, source="ogb", reason="homogeneous OGB schema; full feature hashing is skipped by resource guard"))
    output = Path(args.output)
    write_csv(output, rows, FIELDS)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
