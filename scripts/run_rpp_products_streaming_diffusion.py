from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.eval.status import exception_status
from shadow_hgc.features.streaming_diffusion import compute_streaming_diffusion_blocks
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment
from scripts.run_rpp_common import base_row, write_csv, write_report


def _target_edges(graph) -> torch.Tensor:
    edges = [
        graph.edge_index[relation]
        for relation in graph.relations
        if relation.source_type == graph.target_type and relation.destination_type == graph.target_type
    ]
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.cat(edges, dim=1)


def _diffusion_spec(name: str) -> dict:
    if name == "base":
        return {"steps": (), "include_highpass": False}
    if name == "streaming_diffusion_X0X1":
        return {"steps": (1,), "include_highpass": False}
    if name == "streaming_diffusion_X0X1X2":
        return {"steps": (1, 2), "include_highpass": False}
    if name == "streaming_diffusion_X0X1X2_highpass":
        return {"steps": (1, 2), "include_highpass": True}
    raise ValueError(f"unknown feature variant: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R++ ogbn-products streaming diffusion, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--diffusion-block-dim", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=200000)
    parser.add_argument("--output", default="experiments/tables/products_streaming_diffusion_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/products_streaming_diffusion_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/products_streaming_diffusion_seed42")
    args = parser.parse_args()

    dataset = "ogbn-products"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        graph = load_ogb_node_property_dataset(dataset, download=args.download)
    except Exception as exc:
        path = log_dir / f"{dataset}_load_failed.json"
        payload = {"dataset": dataset, "seed": args.seed, "status": "data_not_available", "reason": str(exc)}
        write_json_summary(path, payload)
        rows = [base_row(path, dataset=dataset, variant="load_failed", summary=payload)]
        write_csv(args.output, rows)
        write_report(args.report_output, title="ogbn-products R++ Streaming Diffusion Summary", rows=rows, csv_path=args.output, previous_best={dataset: 0.5891})
        return

    ratios = [0.02, 0.06, 0.12]
    features = ["base", "streaming_diffusion_X0X1", "streaming_diffusion_X0X1X2", "streaming_diffusion_X0X1X2_highpass"]
    losses = ["sqrt_weighted", "sqrt_weighted_logit_adjusted"]
    edges = _target_edges(graph)
    rows = []
    diffusion_cache: dict[str, dict] = {"base": {}}
    for feature in features:
        if feature == "base":
            continue
        spec = _diffusion_spec(feature)
        out_dir = log_dir / f"{feature}_memmap"
        try:
            start = time.perf_counter()
            result = compute_streaming_diffusion_blocks(
                x_provider=graph.node_features[graph.target_type].to(torch.float32),
                edge_index=edges,
                num_nodes=graph.num_nodes[graph.target_type],
                steps=spec["steps"],
                include_highpass=spec["include_highpass"],
                out_dir=out_dir,
                dtype="float16",
                block_dim=args.diffusion_block_dim,
                edge_chunk_size=args.edge_chunk_size,
                overwrite=True,
            )
            diffusion_cache[feature] = {
                **result.stats,
                "diffusion_precompute_time_s": time.perf_counter() - start,
                "diffusion_backend": result.stats.get("diffusion_backend", "edge_chunk"),
                "diffusion_storage": "fp16_memmap",
                "diffusion_block_paths": {name: str(path) for name, path in result.block_paths.items()},
                "diffusion_block_shapes": {name: list(shape) for name, shape in result.block_shapes.items()},
                "diffusion_status": "completed",
            }
        except Exception as exc:
            diffusion_cache[feature] = {
                "diffusion_status": exception_status(exc),
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            }

    for feature in features:
        for ratio in ratios:
            for loss_type in losses:
                path = log_dir / f"{dataset}_{feature}_{loss_type}_{ratio_slug(ratio)}_seed{args.seed}.json"
                try:
                    if diffusion_cache.get(feature, {}).get("diffusion_status") not in (None, "", "completed"):
                        raise RuntimeError(f"streaming diffusion failed: {diffusion_cache[feature].get('reason')}")
                    if args.skip_existing and path.exists():
                        summary = json.loads(path.read_text(encoding="utf-8"))
                    else:
                        summary = run_shadow_hgc_experiment(
                            graph,
                            output_path=path,
                            method_name="Shadow-HGC-R++",
                            seed=args.seed,
                            epochs=args.epochs,
                            budget_mode="ratio",
                            ratio=ratio,
                            ratio_base="train_target",
                            feature_dim=128,
                            projection_type="random",
                            feature_mode="base",
                            model_type="shadow_fusion",
                            relation_gate=True,
                            loss_type=loss_type,
                            min_proto_per_class=4,
                            budget_alpha=0.5,
                            assignment_chunk_size=4096,
                            inference_edge_chunk_size=args.edge_chunk_size,
                            inference_dst_chunk_size=8192,
                        )
                        summary.update(diffusion_cache.get(feature, {}))
                        summary["feature_mode"] = feature
                        summary["peak_cpu_ram_gb"] = current_cpu_ram_bytes() / 1e9
                        summary["peak_gpu_ram_gb"] = current_gpu_ram_bytes() / 1e9
                        summary["disk_bytes"] = int(summary.get("disk_bytes", 0) or 0) + int(summary.get("diffusion_disk_bytes", 0) or 0)
                        write_json_summary(path, summary)
                    rows.append(base_row(path, dataset=dataset, variant=feature, summary=summary))
                except Exception as exc:
                    payload = {
                        "dataset": dataset,
                        "variant": feature,
                        "seed": args.seed,
                        "ratio": ratio,
                        "loss_type": loss_type,
                        "status": exception_status(exc),
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                        **diffusion_cache.get(feature, {}),
                    }
                    write_json_summary(path, payload)
                    rows.append(base_row(path, dataset=dataset, variant=feature, summary=payload))
    write_csv(args.output, rows)
    write_report(args.report_output, title="ogbn-products R++ Streaming Diffusion Summary", rows=rows, csv_path=args.output, previous_best={dataset: 0.5891})
    print(f"wrote {args.output}")
    print(f"wrote {args.report_output}")


if __name__ == "__main__":
    main()
