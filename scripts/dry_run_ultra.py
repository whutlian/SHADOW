from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.data.edge_stream import run_synthetic_streaming_stress
from shadow_hgc.demand.cache import estimate_ultra_dry_run
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.prototype.budgets import compute_target_budget_from_ratio


DATASET_DEFAULTS = {
    "ogbn-papers100M": {
        "num_train_targets": 1_207_179,
        "num_target_nodes": 111_059_956,
        "num_source_nodes": 111_059_956,
        "num_train_classes": 172,
        "num_relations": 2,
        "active_source_count": 20_000_000,
        "train_train_edges": 120_000_000,
    },
    "mag240m": {
        "num_train_targets": 1_000_000,
        "num_target_nodes": 121_000_000,
        "num_source_nodes": 244_000_000,
        "num_train_classes": 153,
        "num_relations": 3,
        "active_source_count": 30_000_000,
        "train_train_edges": 150_000_000,
    },
}


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


def _dataset_default(args: argparse.Namespace, name: str, fallback: int) -> int:
    if args.dataset in DATASET_DEFAULTS:
        return int(DATASET_DEFAULTS[args.dataset].get(name, fallback))
    return fallback


def _ratio_estimates(args: argparse.Namespace, relations: list[dict], estimate: dict) -> list[dict]:
    rows = []
    target_target_count = sum(1 for relation in relations if relation.get("source_is_target"))
    non_target_count = max(0, len(relations) - target_target_count)
    for ratio in args.ratios:
        budget = compute_target_budget_from_ratio(
            num_train_target_nodes=args.num_train_targets,
            num_train_classes=args.num_train_classes,
            ratio=ratio,
            min_proto_per_class=args.min_proto_per_class,
            max_target_budget=args.max_target_budget,
            rounding=args.budget_rounding,
        )
        effective = int(budget["effective_target_prototypes"])
        tt_shadow = (
            max(args.min_shadow_per_relation, int((args.shadow_ratio_target_target * effective + target_target_count - 1) // max(target_target_count, 1)))
            if target_target_count
            else 0
        )
        nt_shadow = (
            max(args.min_shadow_per_relation, int((args.shadow_ratio_non_target * effective + non_target_count - 1) // max(non_target_count, 1)))
            if non_target_count
            else 0
        )
        shadow_nodes_total = tt_shadow * target_target_count + nt_shadow * non_target_count
        condensed_nodes_total = effective + shadow_nodes_total
        shadow_edges = effective * len(relations)
        skeleton_edges = effective * args.k_s * target_target_count
        condensed_edges_total = shadow_edges + skeleton_edges
        row = {
            **budget,
            "dataset": args.dataset,
            "shadow_ratio_target_target": args.shadow_ratio_target_target,
            "shadow_ratio_non_target": args.shadow_ratio_non_target,
            "min_shadow_per_relation": args.min_shadow_per_relation,
            "shadow_nodes_total": int(shadow_nodes_total),
            "condensed_nodes_total": int(condensed_nodes_total),
            "condensed_edges_total": int(condensed_edges_total),
            "condensed_node_ratio_to_train_target": float(condensed_nodes_total / max(1, args.num_train_targets)),
            "condensed_node_ratio_to_all_task_nodes": float(condensed_nodes_total / max(1, args.num_target_nodes)),
            "demand_cache_GB": float(estimate["demand_cache_bytes"] / 1e9),
            "edge_slice_cache_GB": float(estimate["edge_slice_cache_bytes"] / 1e9),
            "peak_ram_estimate_GB": float(estimate["peak_ram_estimate_bytes"] / 1e9),
            "disk_bytes_estimate_GB": float(estimate["disk_spill_estimate_bytes"] / 1e9),
            "full_edge_scans": int(estimate["total_expected_full_edge_scans"]),
            "cache_all_targets": False,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ultra-scale Shadow-HGC-R-1 cache sizes.")
    parser.add_argument("--output", default="experiments/logs/scaling_stress/dry_run.json")
    parser.add_argument("--dataset", choices=["synthetic", "ogbn-papers100M", "mag240m"], default="synthetic")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--ratios", nargs="*", type=float)
    parser.add_argument("--min-proto-per-class", type=int, default=4)
    parser.add_argument("--max-target-budget", type=int)
    parser.add_argument("--budget-rounding", choices=["nearest", "ceil", "floor"], default="nearest")
    parser.add_argument("--shadow-ratio-target-target", type=float, default=0.5)
    parser.add_argument("--shadow-ratio-non-target", type=float, default=1.0)
    parser.add_argument("--min-shadow-per-relation", type=int, default=8)
    parser.add_argument("--k-s", type=int, default=4)
    parser.add_argument("--num-train-classes", type=int)
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
    if args.dataset != "synthetic":
        defaults = DATASET_DEFAULTS[args.dataset]
        parser_defaults = {
            "num_train_targets": 1_000_000,
            "num_relations": 3,
            "num_target_nodes": 121_000_000,
            "num_source_nodes": 121_000_000,
            "active_source_count": 5_000_000,
            "train_train_edges": 20_000_000,
        }
        for field, default_value in parser_defaults.items():
            if getattr(args, field) == default_value:
                setattr(args, field, int(defaults[field]))
    if args.num_train_classes is None:
        args.num_train_classes = _dataset_default(args, "num_train_classes", 40)
    ratio_values = []
    if args.ratio is not None:
        ratio_values.append(args.ratio)
    if args.ratios:
        ratio_values.extend(args.ratios)
    args.ratios = ratio_values or [0.001]
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
        "dataset": args.dataset,
        "budget_mode": "ratio",
        "ratios": args.ratios,
        "estimate": estimate,
        "ratio_estimates": _ratio_estimates(args, relations, estimate),
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
