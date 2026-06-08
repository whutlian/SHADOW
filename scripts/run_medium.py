from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.baselines.target_coreset import run_target_coreset_baselines
from shadow_hgc.baselines.full_graph_same_backbone import run_full_graph_same_backbone
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.budgeting import make_budget_run_specs
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.eval.tables import build_medium_ablation_rows_from_logs, build_medium_main_rows_from_logs, write_rows_csv
from shadow_hgc.pipeline.ablation import write_skeleton_coverage_figure
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 medium-dataset status table entry point.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--output", default="experiments/tables/medium_main.csv")
    parser.add_argument("--ablation-output", default="experiments/tables/medium_ablation.csv")
    parser.add_argument("--log-dir", default="experiments/logs/medium")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--M-tau", type=int)
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
    parser.add_argument("--k-s", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--projection-type", choices=["raw", "random"], default="random")
    parser.add_argument("--degree-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=["weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted", "sqrt_weighted_logit_adjusted"],
        default="sqrt_weighted",
    )
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp"], default="relation_mlp")
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
    parser.add_argument("--relation-gate", type=_bool_arg, default=False)
    parser.add_argument("--relation-gate-init", type=float, default=1.0)
    parser.add_argument("--skeleton-policy", choices=["fixed_k", "coverage"], default="fixed_k")
    parser.add_argument("--skeleton-coverage", type=float, default=0.65)
    parser.add_argument("--skeleton-k-max", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-proto-per-class", type=int, default=4)
    parser.add_argument("--budget-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-graph-baseline", action="store_true")
    parser.add_argument("--self-only-baseline", action="store_true")
    parser.add_argument("--coreset-baselines", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    ablation_rows = []
    for dataset in args.datasets:
        log_path = Path(args.log_dir) / f"{dataset}.json"
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            payload = {
                "dataset": dataset,
                "method": "Shadow-HGC-R-1",
                "mode": "stage3_status",
                "status": "ogb_data_not_available",
                "reason": str(exc),
                "directed_relations": ["forward", "reverse"],
                "k_s_values": [0, 1, 2, 4, 8],
            }
            write_json_summary(log_path, payload)
            for k_s in [0, 1, 2, 4, 8]:
                ablation_rows.append(
                    {
                        "dataset": dataset,
                        "ablation": "target_target_skeleton",
                        "setting": f"k_s={k_s}",
                        "seed": args.seed,
                        "accuracy": "",
                        "skeleton_coverage_mean": "",
                        "residual_energy_mean": "",
                        "shadow_recon_err_mean": "",
                        "condensation_time": "",
                        "training_time": "",
                        "status": payload["status"],
                    }
                )
            continue

        target_budgets = []
        if args.target_budget is not None:
            target_budgets.append(args.target_budget)
        if args.target_budgets:
            target_budgets.extend(args.target_budgets)
        legacy_budgets = args.M_tau_values or ([] if args.ratios or args.ratio is not None or target_budgets else ([args.M_tau] if args.M_tau is not None else ([200, 400, 800] if dataset == "ogbn-arxiv" else [500, 1000, 2000])))
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
            default_ratios=[0.001, 0.0025, 0.005, 0.01],
            default_target_budgets=[200, 400, 800] if dataset == "ogbn-arxiv" else [500, 1000, 2000],
        )
        if args.full_graph_baseline:
            full_path = Path(args.log_dir) / f"{dataset}_Full-WRL-GNN_seed{args.seed}.json"
            try:
                if not (args.skip_existing and full_path.exists()):
                    run_full_graph_same_backbone(
                        graph,
                        output_path=full_path,
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
            except Exception as exc:
                status = exception_status(exc)
                write_json_summary(full_path, {"dataset": dataset, "method": "Full-WRL-GNN", "status": status, "reason": str(exc)})

        for spec in budget_specs:
            main_path = Path(args.log_dir) / f"{dataset}_{spec.label}.json"
            try:
                if args.skip_existing and main_path.exists():
                    main_summary = json.loads(main_path.read_text(encoding="utf-8"))
                else:
                    main_summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=main_path,
                        seed=args.seed,
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
                    )
            except Exception as exc:
                status = exception_status(exc)
                payload = {
                    "dataset": dataset,
                    "method": "Shadow-HGC-R-1",
                    "mode": "stage3_medium_experiment",
                    "status": status,
                    "reason": str(exc),
                    "budget_mode": spec.budget_mode,
                    "ratio": spec.ratio,
                    "requested_target_budget": spec.target_budget,
                    "directed_relations": [str(relation) for relation in graph.relations],
                    "k_s_values": [0, 1, 2, 4, 8],
                }
                write_json_summary(main_path, payload)
                continue
            if args.self_only_baseline:
                self_path = Path(args.log_dir) / f"{dataset}_Self-Only-MLP_{spec.label}.json"
                try:
                    if not (args.skip_existing and self_path.exists()):
                        run_shadow_hgc_experiment(
                            graph,
                            output_path=self_path,
                            method_name="Self-Only-MLP",
                            seed=args.seed,
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
                            self_only=True,
                        )
                except Exception as exc:
                    status = exception_status(exc)
                    write_json_summary(self_path, {"dataset": dataset, "method": "Self-Only-MLP", "status": status, "reason": str(exc)})
            if args.coreset_baselines:
                run_target_coreset_baselines(
                    graph,
                    seed=args.seed,
                    epochs=args.epochs,
                    M_tau=int(main_summary["effective_target_prototypes"]),
                    feature_dim=args.feature_dim,
                    projection_type=args.projection_type,
                    log_dir=Path(args.log_dir),
                    output_label=spec.label,
                    baseline_match_mode="target_ratio",
                    shadow_condensed_nodes_total=int(main_summary["condensed_nodes_total"]),
                    ratio=spec.ratio,
                    ratio_base=args.ratio_base,
                )
        for k_s in [0, 1, 2, 4, 8]:
            first_spec = budget_specs[0]
            ablation_log_path = Path(args.log_dir) / f"{dataset}_{first_spec.label}_ks{k_s}.json"
            try:
                if args.skip_existing and ablation_log_path.exists():
                    summary = json.loads(ablation_log_path.read_text(encoding="utf-8"))
                else:
                    summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=ablation_log_path,
                        seed=args.seed,
                        epochs=args.epochs,
                        budget_mode=first_spec.budget_mode,
                        ratio=first_spec.ratio,
                        ratio_base=args.ratio_base,
                        target_budget=first_spec.target_budget,
                        max_target_budget=args.max_target_budget,
                        budget_rounding=args.budget_rounding,
                        M_r=args.M_r,
                        k_s=k_s,
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
                    )
                diag_values = [
                    diag
                    for diag in summary["diagnostics"].values()
                    if isinstance(diag, dict) and "ShadowReconErr" in diag
                ]
                ablation_rows.append(
                    {
                        "dataset": dataset,
                        "ablation": "target_target_skeleton",
                        "setting": f"k_s={k_s}",
                        "seed": args.seed,
                        "accuracy": "" if summary["accuracy"] is None else f"{summary['accuracy']:.6f}",
                        "skeleton_coverage_mean": f"{sum(d['SkeletonMassCoverage'] for d in diag_values) / len(diag_values):.6f}",
                        "residual_energy_mean": f"{sum(d['ResidualEnergy'] for d in diag_values) / len(diag_values):.6f}",
                        "shadow_recon_err_mean": f"{sum(d['ShadowReconErr'] for d in diag_values) / len(diag_values):.6f}",
                        "condensation_time": f"{summary['condensation_time']:.6f}",
                        "training_time": f"{summary['training_time']:.6f}",
                        "status": "completed",
                    }
                )
            except Exception as exc:
                status = exception_status(exc)
                write_json_summary(
                    ablation_log_path,
                    {
                        "dataset": dataset,
                        "method": "Shadow-HGC-R-1",
                        "mode": "stage3_medium_ablation",
                        "status": status,
                        "reason": str(exc),
                        "k_s": k_s,
                    },
                )
                ablation_rows.append(
                    {
                        "dataset": dataset,
                        "ablation": "target_target_skeleton",
                        "setting": f"k_s={k_s}",
                        "seed": args.seed,
                        "accuracy": "",
                        "skeleton_coverage_mean": "",
                        "residual_energy_mean": "",
                        "shadow_recon_err_mean": "",
                        "condensation_time": "",
                        "training_time": "",
                        "status": status,
                    }
                )

    output = Path(args.output)
    write_rows_csv(output, build_medium_main_rows_from_logs(args.log_dir))

    ablation_output = Path(args.ablation_output)
    write_rows_csv(ablation_output, build_medium_ablation_rows_from_logs(args.log_dir) or ablation_rows)
    write_skeleton_coverage_figure(
        ablation_rows,
        csv_path="experiments/figures/skeleton_coverage_vs_accuracy_medium.csv",
        svg_path="experiments/figures/skeleton_coverage_vs_accuracy_medium.svg",
    )
    print(f"wrote {output}")
    print(f"wrote {ablation_output}")


if __name__ == "__main__":
    main()
