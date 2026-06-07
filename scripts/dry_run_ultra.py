from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.data.edge_stream import run_synthetic_streaming_stress
from shadow_hgc.demand.cache import estimate_ultra_dry_run
from shadow_hgc.eval.logging import write_json_summary


def _load_relations_arg(value: str) -> list[dict]:
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("relations", payload)
    if not isinstance(payload, list):
        raise ValueError("--relations-json must be a JSON list or an object with a relations list")
    return [dict(item) for item in payload]


def _default_relations(args: argparse.Namespace) -> list[dict]:
    relations: list[dict] = []
    per_relation_active = max(1, args.active_source_count // max(1, args.num_relations))
    per_relation_incident = max(0, args.train_train_edges // max(1, args.num_relations))
    for idx in range(args.num_relations):
        source_is_target = idx == 0
        relations.append(
            {
                "name": f"relation_{idx}",
                "num_edges": per_relation_incident,
                "num_train_target_incident_edges": per_relation_incident,
                "num_train_train_edges": args.train_train_edges if source_is_target else 0,
                "num_active_sources": per_relation_active,
                "num_source_nodes": args.num_source_nodes,
                "num_target_nodes": args.num_target_nodes,
                "source_is_target": source_is_target,
            }
        )
    return relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ultra-scale Shadow-HGC-R-1 cache sizes.")
    parser.add_argument("--output", default="experiments/logs/scaling_stress/dry_run.json")
    parser.add_argument("--num-train-targets", type=int, default=1_000_000)
    parser.add_argument("--num-relations", type=int, default=3)
    parser.add_argument("--num-target-nodes", type=int, default=121_000_000)
    parser.add_argument("--num-source-nodes", type=int, default=121_000_000)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--active-source-count", type=int, default=5_000_000)
    parser.add_argument("--train-train-edges", type=int, default=20_000_000)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--dense-map-budget-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--relations-json", default=None, help="JSON string/path containing per-relation estimates")
    parser.add_argument("--stress", action="store_true", help="also run a synthetic streaming stress cache")
    parser.add_argument("--stress-edges", type=int, default=200_000)
    parser.add_argument("--stress-src", type=int, default=20_000)
    parser.add_argument("--stress-dst", type=int, default=10_000)
    parser.add_argument("--stress-train", type=int, default=1_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    relations = _load_relations_arg(args.relations_json) if args.relations_json else _default_relations(args)

    estimate = estimate_ultra_dry_run(
        num_train_targets=args.num_train_targets,
        feature_dim=args.feature_dim,
        dtype_bytes=args.dtype_bytes,
        relations=relations,
        dense_map_budget_bytes=args.dense_map_budget_bytes,
        num_target_nodes=args.num_target_nodes,
        num_source_nodes=args.num_source_nodes,
    )
    payload = {
        "method": "Shadow-HGC-R-1",
        "mode": "ultra_dry_run",
        "estimate": estimate,
    }
    if args.stress:
        stress_output = args.output.replace(".json", "_stress.json")
        payload["stress"] = run_synthetic_streaming_stress(
            output_path=stress_output,
            num_edges=args.stress_edges,
            num_src_nodes=args.stress_src,
            num_dst_nodes=args.stress_dst,
            num_train_targets=args.stress_train,
            feature_dim=args.feature_dim,
            chunk_size=args.chunk_size,
        )
    write_json_summary(args.output, payload)
    print(f"wrote {args.output}")
    print(f"expected_full_edge_scans={estimate['total_expected_full_edge_scans']}")


if __name__ == "__main__":
    main()
