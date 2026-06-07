from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_hgc.baselines.target_coreset import run_target_coreset_baselines
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.pipeline.ablation import run_ablation_suite, write_skeleton_coverage_figure
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 small-dataset smoke table entry point.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="experiments/tables/small_main.csv")
    parser.add_argument("--ablation-output", default="experiments/tables/small_ablation.csv")
    parser.add_argument("--log-dir", default="experiments/logs/small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--M-tau", type=int, default=32)
    parser.add_argument("--M-tau-values", nargs="*", type=int)
    parser.add_argument("--M-r", type=int, default=8)
    parser.add_argument("--k-s", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--ablation-seed", type=int, default=0)
    parser.add_argument("--figure-csv", default="experiments/figures/skeleton_coverage_vs_accuracy.csv")
    parser.add_argument("--figure-svg", default="experiments/figures/skeleton_coverage_vs_accuracy.svg")
    args = parser.parse_args()

    rows = []
    log_dir = Path(args.log_dir)
    m_tau_values = args.M_tau_values or [args.M_tau]
    for dataset in args.datasets:
        graph = load_processed_small_dataset(dataset)
        for M_tau in m_tau_values:
            acc = []
            for seed in args.seeds:
                log_path = log_dir / f"{dataset}_M{M_tau}_seed{seed}.json"
                summary = run_shadow_hgc_experiment(
                    graph,
                    output_path=log_path,
                    seed=seed,
                    epochs=args.epochs,
                    M_tau=M_tau,
                    M_r=args.M_r,
                    k_s=args.k_s,
                    feature_dim=args.feature_dim,
                )
                acc.append(summary["accuracy"])
            rows.append(
                {
                    "dataset": dataset,
                    "method": "Shadow-HGC-R-1",
                    "mode": "processed_local",
                    "M_tau": M_tau,
                    "seeds": " ".join(str(seed) for seed in args.seeds),
                    "accuracy_mean": f"{statistics.mean(acc):.6f}",
                    "accuracy_std": f"{statistics.pstdev(acc):.6f}",
                    "status": "completed",
                }
            )
            baseline_by_method = {method: [] for method in ["Random-HG", "Herding-HG", "K-Center-HG"]}
            for seed in args.seeds:
                baseline_rows = run_target_coreset_baselines(
                    graph,
                    seed=seed,
                    epochs=args.epochs,
                    M_tau=M_tau,
                    feature_dim=args.feature_dim,
                )
                for baseline_row in baseline_rows:
                    if baseline_row["accuracy"] != "":
                        baseline_by_method[baseline_row["method"]].append(float(baseline_row["accuracy"]))
            for method, baseline_acc in baseline_by_method.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "mode": "target_feature_coreset_baseline",
                        "M_tau": M_tau,
                        "seeds": " ".join(str(seed) for seed in args.seeds),
                        "accuracy_mean": f"{statistics.mean(baseline_acc):.6f}",
                        "accuracy_std": f"{statistics.pstdev(baseline_acc):.6f}",
                        "status": "completed",
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
            )
        )
    ablation_output = Path(args.ablation_output)
    ablation_output.parent.mkdir(parents=True, exist_ok=True)
    with ablation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    write_skeleton_coverage_figure(ablation_rows, csv_path=args.figure_csv, svg_path=args.figure_svg)
    print(f"wrote {output}")
    print(f"wrote {ablation_output}")
    print(f"wrote {args.figure_csv}")
    print(f"wrote {args.figure_svg}")


if __name__ == "__main__":
    main()
