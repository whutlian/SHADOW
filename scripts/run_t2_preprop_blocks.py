from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import (
    ALL_T2_DATASETS,
    MEDIUM_DATASETS,
    build_t2_block_groups,
    load_t2_graph,
    markdown_table,
    split_train_valid,
    write_csv,
    write_json,
)
from shadow_hgc.fullgraph.metapath_specs import available_metapath_specs
from shadow_hgc.preprop import PrepropBlockSpec, compute_preprop_blocks


FIELDS = [
    "dataset",
    "status",
    "manifest_dir",
    "num_blocks",
    "block_names",
    "total_cache_bytes",
    "full_edge_scans",
    "edge_chunk_size",
    "dst_chunk_size",
    "block_dim",
    "uses_memmap",
    "uses_logits_as_input",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_bounded_edges",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "wall_time_s",
    "reason",
]


def _feature_provider(graph) -> dict[str, torch.Tensor]:
    provider = {name: value.to(torch.float32) for name, value in graph.node_features.items()}
    if graph.target_type not in provider:
        provider[graph.target_type] = torch.zeros(graph.num_nodes[graph.target_type], 16, dtype=torch.float32)
    return provider


def _specs_for_graph(graph, train_rows: torch.Tensor) -> list[PrepropBlockSpec]:
    specs: list[PrepropBlockSpec] = [PrepropBlockSpec.self_block(name="X0", train_rows=train_rows.detach().cpu().tolist())]
    for relation in graph.relations:
        if relation.destination_type == graph.target_type and relation.source_type in graph.node_features:
            specs.append(
                PrepropBlockSpec.typed_feature(
                    name=f"X1_{relation.relation_name}",
                    relation=relation,
                    train_rows=train_rows.detach().cpu().tolist(),
                )
            )
    if graph.dataset_name in {"acm", "dblp", "imdb"}:
        metapaths, _ = available_metapath_specs(graph.dataset_name, graph.relations, graph.target_type)
        for name, path in metapaths.items():
            specs.append(PrepropBlockSpec.metapath_feature(name=f"metapath_{name}", path_schema=path, train_rows=train_rows.detach().cpu().tolist()))
    return specs


def run_preprop_dataset(dataset: str, args) -> dict[str, Any]:
    if dataset == "ogbn-products" and not args.run_products_full:
        return {
            "dataset": dataset,
            "status": "blocked_resource_guard",
            "manifest_dir": "",
            "num_blocks": 0,
            "block_names": "[]",
            "total_cache_bytes": 0,
            "full_edge_scans": 0,
            "edge_chunk_size": args.edge_chunk_size,
            "dst_chunk_size": args.dst_chunk_size,
            "block_dim": args.medium_block_dim,
            "uses_memmap": True,
            "uses_logits_as_input": False,
            "uses_e_by_d_materialization": False,
            "uses_dense_p2": False,
            "uses_bounded_edges": False,
            "reason": "products full preprop skipped locally; dry-run covers resources",
        }
    graph = load_t2_graph(dataset)
    train_rows, _ = split_train_valid(graph, seed=args.seed)
    block_dim = args.medium_block_dim if dataset in MEDIUM_DATASETS else args.block_dim
    manifest_dir = Path(args.output_dir) / dataset
    manifest = compute_preprop_blocks(
        dataset_name=dataset,
        target_type=graph.target_type,
        block_specs=_specs_for_graph(graph, train_rows),
        feature_provider=_feature_provider(graph),
        edge_store=graph.edge_index,
        output_dir=str(manifest_dir),
        dtype=args.dtype,
        block_dim=block_dim,
        edge_chunk_size=args.edge_chunk_size,
        dst_chunk_size=args.dst_chunk_size,
        use_memmap=True,
        seed=args.seed,
    )
    return {
        "dataset": dataset,
        "status": "completed",
        "manifest_dir": str(manifest_dir),
        "num_blocks": len(manifest.blocks),
        "block_names": json.dumps([block.name for block in manifest.blocks], sort_keys=True),
        "total_cache_bytes": manifest.total_cache_bytes,
        "full_edge_scans": manifest.full_edge_scans,
        "edge_chunk_size": manifest.edge_chunk_size,
        "dst_chunk_size": manifest.dst_chunk_size,
        "block_dim": manifest.block_dim,
        "uses_memmap": manifest.uses_memmap,
        "uses_logits_as_input": manifest.uses_logits_as_input,
        "uses_e_by_d_materialization": manifest.uses_e_by_d_materialization,
        "uses_dense_p2": manifest.uses_dense_p2,
        "uses_bounded_edges": manifest.uses_bounded_edges,
        "peak_cpu_ram_gb": manifest.peak_cpu_ram_gb,
        "peak_gpu_ram_gb": manifest.peak_gpu_ram_gb,
        "wall_time_s": manifest.wall_time_s,
        "reason": "completed",
    }


def _write_report(rows: list[dict[str, Any]], output: Path, report: Path) -> None:
    lines = [
        "# T2-SFT-NL Preprop Manifest Summary",
        "",
        "Every completed manifest records `uses_logits=false` per block and `uses_logits_as_input=false` at run level.",
        "",
        *markdown_table(rows, ["dataset", "status", "num_blocks", "total_cache_bytes", "full_edge_scans", "uses_logits_as_input", "uses_e_by_d_materialization", "uses_dense_p2", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2 no-logits preprop block manifests.")
    parser.add_argument("--datasets", nargs="+", default=ALL_T2_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--medium-block-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--run-products-full", action="store_true")
    parser.add_argument("--output-dir", default="experiments/preprop/t2_seed42")
    parser.add_argument("--output", default="experiments/tables/t2_preprop_manifest_index_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t2_preprop_summary.md")
    args = parser.parse_args()
    rows = [run_preprop_dataset(dataset, args) for dataset in args.datasets]
    output = write_csv(args.output, rows, FIELDS)
    write_json(Path(args.output).with_suffix(".json"), {"rows": rows})
    _write_report(rows, output, Path(args.report))
    print(json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
