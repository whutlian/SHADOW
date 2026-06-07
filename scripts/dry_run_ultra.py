from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.data.edge_stream import run_synthetic_streaming_stress
from shadow_hgc.demand.cache import estimate_ultra_dry_run
from shadow_hgc.eval.logging import write_json_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ultra-scale Shadow-HGC-R-1 cache sizes.")
    parser.add_argument("--output", default="experiments/logs/scaling_stress/dry_run.json")
    parser.add_argument("--num-train-targets", type=int, default=1_000_000)
    parser.add_argument("--num-relations", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--active-source-count", type=int, default=5_000_000)
    parser.add_argument("--train-train-edges", type=int, default=20_000_000)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--stress", action="store_true", help="also run a synthetic streaming stress cache")
    parser.add_argument("--stress-edges", type=int, default=200_000)
    parser.add_argument("--stress-src", type=int, default=20_000)
    parser.add_argument("--stress-dst", type=int, default=10_000)
    parser.add_argument("--stress-train", type=int, default=1_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    estimate = estimate_ultra_dry_run(
        num_train_targets=args.num_train_targets,
        num_relations=args.num_relations,
        feature_dim=args.feature_dim,
        active_source_count=args.active_source_count,
        train_train_edges=args.train_train_edges,
        dtype_bytes=args.dtype_bytes,
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
    print(f"expected_full_edge_scans={estimate['expected_full_edge_scans']}")


if __name__ == "__main__":
    main()
