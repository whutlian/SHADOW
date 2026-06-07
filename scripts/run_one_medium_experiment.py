from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one medium Shadow-HGC-R-1 experiment.")
    parser.add_argument("--dataset", required=True, choices=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-name", default="Shadow-HGC-R-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--M-tau", type=int, required=True)
    parser.add_argument("--M-r", type=int)
    parser.add_argument("--k-s", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--projection-type", choices=["raw", "random"], default="random")
    parser.add_argument("--degree-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted"],
        default="sqrt_weighted",
    )
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp"], default="relation_mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-proto-per-class", type=int, default=4)
    parser.add_argument("--shadow-mode", choices=["virtual_demand_shadow", "real_source_centroid", "private_shadow"], default="virtual_demand_shadow")
    parser.add_argument("--self-only", action="store_true")
    args = parser.parse_args()

    graph = load_ogb_node_property_dataset(args.dataset, download=False)
    summary = run_shadow_hgc_experiment(
        graph,
        output_path=args.output,
        method_name=args.method_name,
        seed=args.seed,
        epochs=args.epochs,
        M_tau=args.M_tau,
        M_r=args.M_r,
        k_s=args.k_s,
        feature_dim=args.feature_dim,
        projection_type=args.projection_type,
        degree_scale=args.degree_scale,
        loss_type=args.loss_type,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_proto_per_class=args.min_proto_per_class,
        shadow_mode=args.shadow_mode,
        self_only=args.self_only,
    )
    print(f"wrote {args.output}")
    print(f"accuracy={summary['accuracy']:.6f} macro_f1={summary['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
