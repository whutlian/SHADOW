from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.ablation import write_skeleton_coverage_figure
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 medium-dataset status table entry point.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--output", default="experiments/tables/medium_main.csv")
    parser.add_argument("--ablation-output", default="experiments/tables/medium_ablation.csv")
    parser.add_argument("--log-dir", default="experiments/logs/medium")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--M-tau", type=int, default=64)
    parser.add_argument("--M-r", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
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
            rows.append(
                {
                    "dataset": dataset,
                    "method": "Shadow-HGC-R-1",
                    "status": payload["status"],
                    "accuracy": "",
                    "condensation_time": "",
                    "training_time": "",
                    "inference_time": "",
                }
            )
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

        try:
            main_summary = run_shadow_hgc_experiment(
                graph,
                output_path=log_path,
                seed=args.seed,
                epochs=args.epochs,
                M_tau=args.M_tau,
                M_r=args.M_r,
                k_s=2,
                feature_dim=args.feature_dim,
            )
        except Exception as exc:
            status = exception_status(exc)
            payload = {
                "dataset": dataset,
                "method": "Shadow-HGC-R-1",
                "mode": "stage3_medium_experiment",
                "status": status,
                "reason": str(exc),
                "directed_relations": [str(relation) for relation in graph.relations],
                "k_s_values": [0, 1, 2, 4, 8],
            }
            write_json_summary(log_path, payload)
            rows.append(
                {
                    "dataset": dataset,
                    "method": "Shadow-HGC-R-1",
                    "status": status,
                    "accuracy": "",
                    "condensation_time": "",
                    "training_time": "",
                    "inference_time": "",
                }
            )
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
                        "status": status,
                    }
                )
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": "Shadow-HGC-R-1",
                "status": "completed",
                "accuracy": "" if main_summary["accuracy"] is None else f"{main_summary['accuracy']:.6f}",
                "condensation_time": f"{main_summary['condensation_time']:.6f}",
                "training_time": f"{main_summary['training_time']:.6f}",
                "inference_time": f"{main_summary['inference_time']:.6f}",
            }
        )
        for k_s in [0, 1, 2, 4, 8]:
            ablation_log_path = Path(args.log_dir) / f"{dataset}_ks{k_s}.json"
            try:
                summary = run_shadow_hgc_experiment(
                    graph,
                    output_path=ablation_log_path,
                    seed=args.seed,
                    epochs=args.epochs,
                    M_tau=args.M_tau,
                    M_r=args.M_r,
                    k_s=k_s,
                    feature_dim=args.feature_dim,
                )
                diag_values = list(summary["diagnostics"].values())
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ablation_output = Path(args.ablation_output)
    ablation_output.parent.mkdir(parents=True, exist_ok=True)
    with ablation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    write_skeleton_coverage_figure(
        ablation_rows,
        csv_path="experiments/figures/skeleton_coverage_vs_accuracy_medium.csv",
        svg_path="experiments/figures/skeleton_coverage_vs_accuracy_medium.svg",
    )
    print(f"wrote {output}")
    print(f"wrote {ablation_output}")


if __name__ == "__main__":
    main()
