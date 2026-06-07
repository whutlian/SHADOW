from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.baselines.full_graph_same_backbone import run_full_graph_same_backbone
from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline.toy import run_toy_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy Shadow-HGC-R-1 pipeline.")
    parser.add_argument("--output", default="experiments/logs/toy/summary.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--M-tau", type=int, default=4)
    parser.add_argument("--M-r", type=int)
    parser.add_argument("--k-s", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--projection-type", choices=["raw", "random"], default="random")
    parser.add_argument("--degree-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted"],
        default="weighted",
    )
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp"], default="relation_linear")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-proto-per-class", type=int, default=1)
    parser.add_argument("--private-shadow", action="store_true")
    parser.add_argument("--self-only", action="store_true")
    parser.add_argument("--full-graph-same-backbone", action="store_true")
    args = parser.parse_args()
    if args.full_graph_same_backbone:
        graph = build_toy_graph(seed=args.seed)
        summary = run_full_graph_same_backbone(
            graph,
            output_path=args.output,
            seed=args.seed,
            epochs=args.epochs,
            feature_dim=args.feature_dim,
            projection_type=args.projection_type,
            degree_scale=args.degree_scale,
            model_type=args.model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_type=args.loss_type,
        )
        print(f"wrote {args.output}")
        print(f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}")
        return
    summary = run_toy_experiment(
        output_path=args.output,
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
        shadow_mode="private_shadow" if args.private_shadow else "virtual_demand_shadow",
        self_only=args.self_only,
    )
    print(f"wrote {args.output}")
    print(f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
