from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.baselines.full_graph_same_backbone import run_full_graph_same_backbone
from shadow_hgc.baselines.target_coreset import run_target_coreset_baselines
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.tables import build_small_main_rows_from_logs, write_rows_csv
from shadow_hgc.pipeline.ablation import run_ablation_suite, write_skeleton_coverage_figure
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _done(path: Path, *, skip_existing: bool) -> bool:
    return bool(skip_existing and path.exists())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 small-dataset smoke table entry point.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="experiments/tables/small_main.csv")
    parser.add_argument("--ablation-output", default="experiments/tables/small_ablation.csv")
    parser.add_argument("--log-dir", default="experiments/logs/small")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--M-tau", type=int, default=32)
    parser.add_argument("--M-tau-values", nargs="*", type=int)
    parser.add_argument("--M-r", type=int)
    parser.add_argument("--k-s", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--projection-type", choices=["raw", "random"], default="raw")
    parser.add_argument("--degree-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted"],
        default="clipped",
    )
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp"], default="relation_mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-proto-per-class", type=int, default=4)
    parser.add_argument("--budget-alpha", type=float, default=0.5)
    parser.add_argument("--skip-full-graph-baseline", action="store_true")
    parser.add_argument("--skip-coreset-baselines", action="store_true")
    parser.add_argument("--skip-self-only-baseline", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ablation-seed", type=int, default=0)
    parser.add_argument("--figure-csv", default="experiments/figures/skeleton_coverage_vs_accuracy.csv")
    parser.add_argument("--figure-svg", default="experiments/figures/skeleton_coverage_vs_accuracy.svg")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    m_tau_values = args.M_tau_values or [32, 64, 128]
    for dataset in args.datasets:
        graph = load_processed_small_dataset(dataset)
        if not args.skip_full_graph_baseline:
            for seed in args.seeds:
                output_path = log_dir / f"{dataset}_Full-WRL-GNN_seed{seed}.json"
                if _done(output_path, skip_existing=args.skip_existing):
                    continue
                run_full_graph_same_backbone(
                    graph,
                    output_path=output_path,
                    seed=seed,
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
        for M_tau in m_tau_values:
            for seed in args.seeds:
                log_path = log_dir / f"{dataset}_M{M_tau}_seed{seed}.json"
                if not _done(log_path, skip_existing=args.skip_existing):
                    run_shadow_hgc_experiment(
                        graph,
                        output_path=log_path,
                        seed=seed,
                        epochs=args.epochs,
                        M_tau=M_tau,
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
                        budget_alpha=args.budget_alpha,
                    )
                if not args.skip_self_only_baseline:
                    self_path = log_dir / f"{dataset}_Self-Only-MLP_M{M_tau}_seed{seed}.json"
                    if _done(self_path, skip_existing=args.skip_existing):
                        continue
                    run_shadow_hgc_experiment(
                        graph,
                        output_path=self_path,
                        method_name="Self-Only-MLP",
                        seed=seed,
                        epochs=args.epochs,
                        M_tau=M_tau,
                        M_r=args.M_r,
                        k_s=0,
                        feature_dim=args.feature_dim,
                        projection_type=args.projection_type,
                        degree_scale=args.degree_scale,
                        loss_type=args.loss_type,
                        model_type="relation_mlp",
                        hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        min_proto_per_class=args.min_proto_per_class,
                        budget_alpha=args.budget_alpha,
                        self_only=True,
                    )
            if not args.skip_coreset_baselines:
                for seed in args.seeds:
                    coreset_paths = [
                        log_dir / f"{dataset}_{method}_M{M_tau}_seed{seed}.json"
                        for method in ["Random_HG", "Herding_HG", "K_Center_HG"]
                    ]
                    if args.skip_existing and all(path.exists() for path in coreset_paths):
                        continue
                    run_target_coreset_baselines(
                        graph,
                        seed=seed,
                        epochs=args.epochs,
                        M_tau=M_tau,
                        feature_dim=args.feature_dim,
                        projection_type=args.projection_type,
                        log_dir=log_dir,
                    )

    write_rows_csv(args.output, build_small_main_rows_from_logs(log_dir))
    ablation_rows = []
    for dataset in args.datasets:
        graph = load_processed_small_dataset(dataset)
        ablation_rows.extend(
            run_ablation_suite(
                graph,
                log_dir=log_dir / "ablations",
                seed=args.ablation_seed,
                epochs=args.epochs,
                M_tau=m_tau_values[0],
                M_r=args.M_r,
                k_s=args.k_s,
                feature_dim=args.feature_dim,
                projection_type=args.projection_type,
                degree_scale=args.degree_scale,
                min_proto_per_class=args.min_proto_per_class,
                budget_alpha=args.budget_alpha,
                loss_type=args.loss_type,
                model_type=args.model,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                skip_existing=args.skip_existing,
            )
        )
    ablation_output = Path(args.ablation_output)
    ablation_output.parent.mkdir(parents=True, exist_ok=True)
    with ablation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    write_skeleton_coverage_figure(ablation_rows, csv_path=args.figure_csv, svg_path=args.figure_svg)
    print(f"wrote {args.output}")
    print(f"wrote {ablation_output}")
    print(f"wrote {args.figure_csv}")
    print(f"wrote {args.figure_svg}")


if __name__ == "__main__":
    main()
