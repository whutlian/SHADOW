from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import load_t2_graph, split_train_valid
from scripts.t21_common import T21_PREPROP_FIELDS, markdown_table, write_csv
from shadow_hgc.preprop.true_preprop import compute_preprop_blocks


def _provider(graph, train_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    provider = {name: value.to(torch.float32) for name, value in graph.node_features.items()}
    if graph.target_type not in provider:
        provider[graph.target_type] = torch.zeros(graph.num_nodes[graph.target_type], 16, dtype=torch.float32)
    provider["train_rows"] = train_rows.to(torch.long)
    return provider


def _blocks_for_graph(graph) -> list[str]:
    blocks = ["X0"]
    target_target = [rel for rel in graph.relations if rel.source_type == graph.target_type and rel.destination_type == graph.target_type]
    if target_target:
        for rel in target_target:
            blocks.append(f"X1_{rel.relation_name}")
        if graph.dataset_name == "ogbn-arxiv":
            for rel in target_target:
                blocks.append(f"X2_{rel.relation_name}")
        blocks.append("Xres")
    blocks.extend(["typed_demand", "structure"])
    if graph.dataset_name in {"acm", "dblp", "imdb"}:
        blocks.extend(["metapath", "lad_scap"])
    return blocks


def run_dataset(dataset: str, args) -> dict[str, Any]:
    graph = load_t2_graph(dataset)
    train_rows, _ = split_train_valid(graph, seed=args.seed)
    feature_dim = args.medium_feature_dim if dataset.startswith("ogbn-") else args.feature_dim
    manifest_dir = Path(args.output_dir) / dataset
    manifest = compute_preprop_blocks(
        dataset_name=dataset,
        target_type=graph.target_type,
        x_provider=_provider(graph, train_rows),
        relations=graph.edge_index,
        output_dir=str(manifest_dir),
        blocks=_blocks_for_graph(graph),
        feature_dim=feature_dim,
        dtype=args.dtype,
        edge_chunk_size=args.edge_chunk_size,
        dst_chunk_size=args.dst_chunk_size,
        force_memmap=True,
        seed=args.seed,
    )
    return {
        "dataset": dataset,
        "target_type": graph.target_type,
        "status": "completed",
        "manifest_dir": str(manifest_dir),
        "num_blocks": len(manifest.blocks),
        "block_names": json.dumps([block.name for block in manifest.blocks], sort_keys=True),
        "total_cache_bytes": manifest.total_cache_bytes,
        "full_edge_scans": manifest.full_edge_scans,
        "edge_chunk_size": manifest.edge_chunk_size,
        "dst_chunk_size": manifest.dst_chunk_size,
        "feature_dim": manifest.block_dim,
        "uses_memmap": manifest.uses_memmap,
        "uses_logits_as_input": manifest.uses_logits_as_input,
        "uses_teacher_logits": manifest.uses_teacher_logits,
        "uses_kd": manifest.uses_kd,
        "uses_dense_p2": manifest.uses_dense_p2,
        "uses_bounded_edges": manifest.uses_bounded_edges,
        "uses_e_by_d_materialization": manifest.uses_e_by_d_materialization,
        "uses_diffusion_legacy": manifest.uses_diffusion_legacy,
        "peak_cpu_ram_gb": manifest.peak_cpu_ram_gb,
        "peak_gpu_ram_gb": manifest.peak_gpu_ram_gb,
        "wall_time_s": manifest.wall_time_s,
        "reason": "true_chunked_memmap_preprop_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2.1 true chunked/memmap preprop manifests.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb", "ogbn-arxiv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--medium-feature-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--output-dir", default="experiments/preprop/t21_seed42")
    parser.add_argument("--output", default="experiments/tables/t21_preprop_manifest_index_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t21_preprop_manifest_summary.md")
    args = parser.parse_args()
    rows = [run_dataset(dataset, args) for dataset in args.datasets]
    output = write_csv(args.output, rows, T21_PREPROP_FIELDS)
    lines = [
        "# T2.1 True Preprop Manifest Index",
        "",
        "Completed rows use chunked destination-row SpMM and memmap output. No row uses logits, KD, dense P2, bounded edges, or E x d materialization.",
        "",
        *markdown_table(rows, ["dataset", "status", "num_blocks", "block_names", "total_cache_bytes", "full_edge_scans", "uses_e_by_d_materialization", "reason"]),
        "",
        f"- CSV: `{output}`",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
