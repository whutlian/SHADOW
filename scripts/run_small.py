from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.baselines.full_graph_same_backbone import run_full_graph_same_backbone
from shadow_hgc.baselines.target_coreset import run_target_coreset_baselines
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.budgeting import make_budget_run_specs
from shadow_hgc.eval.tables import build_small_main_rows_from_logs, write_rows_csv
from shadow_hgc.pipeline.ablation import run_ablation_suite, write_skeleton_coverage_figure
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _diffusion_steps(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


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
    parser.add_argument("--target-budget", type=int)
    parser.add_argument("--target-budgets", nargs="*", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--ratios", nargs="*", type=float)
    parser.add_argument("--budget-mode", choices=["ratio", "count"])
    parser.add_argument("--ratio-base", choices=["train_target", "all_target"], default="train_target")
    parser.add_argument("--max-target-budget", type=int)
    parser.add_argument("--budget-rounding", choices=["nearest", "ceil", "floor"], default="nearest")
    parser.add_argument("--M-r", type=int)
    parser.add_argument("--shadow-ratio-target-target", type=float, default=0.5)
    parser.add_argument("--shadow-ratio-non-target", type=float, default=1.0)
    parser.add_argument("--min-shadow-per-relation", type=int, default=8)
    parser.add_argument("--k-s", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--projection-type", choices=["raw", "random"], default="raw")
    parser.add_argument("--degree-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted", "sqrt_weighted_logit_adjusted"],
        default="clipped",
    )
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp", "shadow_fusion"], default="relation_mlp")
    parser.add_argument("--shadow-policy", choices=["fixed", "rank_adaptive"], default="fixed")
    parser.add_argument("--shadow-min-per-relation", type=int, default=8)
    parser.add_argument("--shadow-max-multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive-b", type=_bool_arg, default=False)
    parser.add_argument("--b-max", type=int, default=4)
    parser.add_argument("--rank-diagnostic-k", type=int, default=64)
    parser.add_argument("--feature-mode", choices=["base", "diffusion", "metapath", "diffusion_metapath"], default="base")
    parser.add_argument("--diffusion-steps", type=_diffusion_steps, default=(1,))
    parser.add_argument("--include-highpass", type=_bool_arg, default=False)
    parser.add_argument("--metapath-signature", type=_bool_arg, default=False)
    parser.add_argument("--metapath-model-input", type=_bool_arg, default=False)
    parser.add_argument("--multiscale-dim", type=int, default=128)
    parser.add_argument("--block-norm", choices=["none", "standardize", "l2", "standardize_l2"], default="none")
    parser.add_argument("--block-gate", type=_bool_arg, default=False)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--relation-gate", type=_bool_arg, default=False)
    parser.add_argument("--relation-gate-init", type=float, default=1.0)
    parser.add_argument("--skeleton-policy", choices=["fixed_k", "coverage"], default="fixed_k")
    parser.add_argument("--skeleton-coverage", type=float, default=0.65)
    parser.add_argument("--skeleton-k-max", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ratio-mode", choices=["target_only", "total_nodes"], default="target_only")
    parser.add_argument("--shadow-total-budget", type=int)
    parser.add_argument("--rank-adaptive-global-cap", type=_bool_arg, default=False)
    parser.add_argument("--max-total-condensed-ratio", type=float)
    parser.add_argument("--assignment-chunk-size", type=int)
    parser.add_argument("--inference-dst-chunk-size", type=int)
    parser.add_argument("--min-proto-per-class", type=int, default=4)
    parser.add_argument("--budget-alpha", type=float, default=0.5)
    parser.add_argument("--skip-full-graph-baseline", action="store_true")
    parser.add_argument("--skip-coreset-baselines", action="store_true")
    parser.add_argument("--skip-self-only-baseline", action="store_true")
    parser.add_argument(
        "--baseline-match-modes",
        nargs="+",
        choices=["target_ratio", "total_condensed_nodes"],
        default=["target_ratio", "total_condensed_nodes"],
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ablation-seed", type=int, default=0)
    parser.add_argument("--figure-csv", default="experiments/figures/skeleton_coverage_vs_accuracy.csv")
    parser.add_argument("--figure-svg", default="experiments/figures/skeleton_coverage_vs_accuracy.svg")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    target_budgets = []
    if args.target_budget is not None:
        target_budgets.append(args.target_budget)
    if args.target_budgets:
        target_budgets.extend(args.target_budgets)
    legacy_budgets = args.M_tau_values or ([] if args.ratios or args.ratio is not None or target_budgets else [args.M_tau])
    ratios = []
    if args.ratio is not None:
        ratios.append(args.ratio)
    if args.ratios:
        ratios.extend(args.ratios)
    budget_specs = make_budget_run_specs(
        budget_mode=args.budget_mode,
        ratios=ratios,
        target_budgets=target_budgets,
        legacy_target_budgets=legacy_budgets,
        default_ratios=[0.005, 0.01, 0.02, 0.05],
        default_target_budgets=[32, 64, 128],
    )
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
        for spec in budget_specs:
            for seed in args.seeds:
                log_path = log_dir / f"{dataset}_{spec.label}_seed{seed}.json"
                if _done(log_path, skip_existing=args.skip_existing):
                    main_summary = json.loads(log_path.read_text(encoding="utf-8"))
                else:
                    main_summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=log_path,
                        seed=seed,
                        epochs=args.epochs,
                        budget_mode=spec.budget_mode,
                        ratio=spec.ratio,
                        ratio_base=args.ratio_base,
                        target_budget=spec.target_budget,
                        max_target_budget=args.max_target_budget,
                        budget_rounding=args.budget_rounding,
                        M_r=args.M_r,
                        k_s=args.k_s,
                        feature_dim=args.feature_dim,
                        projection_type=args.projection_type,
                        degree_scale=args.degree_scale,
                        loss_type=args.loss_type,
                        logit_adjustment_tau=args.logit_adjustment_tau,
                        model_type=args.model,
                        shadow_policy=args.shadow_policy,
                        shadow_min_per_relation=args.shadow_min_per_relation,
                        shadow_max_multiplier=args.shadow_max_multiplier,
                        adaptive_b=args.adaptive_b,
                        b_max=args.b_max,
                        rank_diagnostic_k=args.rank_diagnostic_k,
                        feature_mode=args.feature_mode,
                        diffusion_steps=args.diffusion_steps,
                        include_highpass=args.include_highpass,
                        metapath_signature=args.metapath_signature,
                        metapath_model_input=args.metapath_model_input,
                        multiscale_dim=args.multiscale_dim,
                        block_norm=args.block_norm,
                        block_gate=args.block_gate,
                        block_dropout=args.block_dropout,
                        relation_gate=args.relation_gate,
                        relation_gate_init=args.relation_gate_init,
                        skeleton_policy=args.skeleton_policy,
                        skeleton_coverage=args.skeleton_coverage,
                        skeleton_k_max=args.skeleton_k_max,
                        hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        min_proto_per_class=args.min_proto_per_class,
                        budget_alpha=args.budget_alpha,
                        shadow_target_target_ratio=args.shadow_ratio_target_target,
                        shadow_non_target_ratio=args.shadow_ratio_non_target,
                        min_shadows_per_relation=args.min_shadow_per_relation,
                        ratio_mode=args.ratio_mode,
                        shadow_total_budget=args.shadow_total_budget,
                        rank_adaptive_global_cap=args.rank_adaptive_global_cap,
                        max_total_condensed_ratio=args.max_total_condensed_ratio,
                        assignment_chunk_size=args.assignment_chunk_size,
                        inference_dst_chunk_size=args.inference_dst_chunk_size,
                    )
                if not args.skip_self_only_baseline:
                    self_path = log_dir / f"{dataset}_Self-Only-MLP_{spec.label}_seed{seed}.json"
                    if _done(self_path, skip_existing=args.skip_existing):
                        continue
                    run_shadow_hgc_experiment(
                        graph,
                        output_path=self_path,
                        method_name="Self-Only-MLP",
                        seed=seed,
                        epochs=args.epochs,
                        budget_mode=spec.budget_mode,
                        ratio=spec.ratio,
                        ratio_base=args.ratio_base,
                        target_budget=spec.target_budget,
                        max_target_budget=args.max_target_budget,
                        budget_rounding=args.budget_rounding,
                        M_r=args.M_r,
                        k_s=0,
                        feature_dim=args.feature_dim,
                        projection_type=args.projection_type,
                        degree_scale=args.degree_scale,
                        loss_type=args.loss_type,
                        logit_adjustment_tau=args.logit_adjustment_tau,
                        model_type="relation_mlp",
                        shadow_policy=args.shadow_policy,
                        shadow_min_per_relation=args.shadow_min_per_relation,
                        shadow_max_multiplier=args.shadow_max_multiplier,
                        adaptive_b=args.adaptive_b,
                        b_max=args.b_max,
                        rank_diagnostic_k=args.rank_diagnostic_k,
                        feature_mode=args.feature_mode,
                        diffusion_steps=args.diffusion_steps,
                        include_highpass=args.include_highpass,
                        metapath_signature=args.metapath_signature,
                        metapath_model_input=args.metapath_model_input,
                        multiscale_dim=args.multiscale_dim,
                        block_norm=args.block_norm,
                        block_gate=args.block_gate,
                        block_dropout=args.block_dropout,
                        relation_gate=args.relation_gate,
                        relation_gate_init=args.relation_gate_init,
                        skeleton_policy=args.skeleton_policy,
                        skeleton_coverage=args.skeleton_coverage,
                        skeleton_k_max=args.skeleton_k_max,
                        hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        min_proto_per_class=args.min_proto_per_class,
                        budget_alpha=args.budget_alpha,
                        shadow_target_target_ratio=args.shadow_ratio_target_target,
                        shadow_non_target_ratio=args.shadow_ratio_non_target,
                        min_shadows_per_relation=args.min_shadow_per_relation,
                        ratio_mode=args.ratio_mode,
                        shadow_total_budget=args.shadow_total_budget,
                        rank_adaptive_global_cap=args.rank_adaptive_global_cap,
                        max_total_condensed_ratio=args.max_total_condensed_ratio,
                        assignment_chunk_size=args.assignment_chunk_size,
                        inference_dst_chunk_size=args.inference_dst_chunk_size,
                        self_only=True,
                    )
                if not args.skip_coreset_baselines:
                    for match_mode in args.baseline_match_modes:
                        baseline_budget = (
                            int(main_summary["effective_target_prototypes"])
                            if match_mode == "target_ratio"
                            else int(main_summary["condensed_nodes_total"])
                        )
                        coreset_paths = [
                            log_dir / f"{dataset}_{method}_{match_mode}_{spec.label}_seed{seed}.json"
                            for method in ["Random_HG", "Herding_HG", "K_Center_HG"]
                        ]
                        if args.skip_existing and all(path.exists() for path in coreset_paths):
                            continue
                        run_target_coreset_baselines(
                            graph,
                            seed=seed,
                            epochs=args.epochs,
                            M_tau=baseline_budget,
                            feature_dim=args.feature_dim,
                            projection_type=args.projection_type,
                            log_dir=log_dir,
                            output_label=spec.label,
                            baseline_match_mode=match_mode,
                            shadow_condensed_nodes_total=int(main_summary["condensed_nodes_total"]),
                            ratio=spec.ratio,
                            ratio_base=args.ratio_base,
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
                M_tau=budget_specs[0].target_budget,
                budget_mode=budget_specs[0].budget_mode,
                ratio=budget_specs[0].ratio,
                ratio_base=args.ratio_base,
                target_budget=budget_specs[0].target_budget,
                max_target_budget=args.max_target_budget,
                budget_rounding=args.budget_rounding,
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
