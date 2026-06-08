from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _variant_config(name: str) -> dict:
    if name == "base":
        return {"feature_mode": "base", "skeleton_policy": "fixed_k", "k_s": 4, "shadow_policy": "fixed", "adaptive_b": False}
    if name == "diffusion_X0X1":
        return {"feature_mode": "diffusion", "diffusion_steps": (1,), "include_highpass": False, "skeleton_policy": "fixed_k", "k_s": 4}
    if name == "diffusion_X0X1X2":
        return {"feature_mode": "diffusion", "diffusion_steps": (1, 2), "include_highpass": False, "skeleton_policy": "fixed_k", "k_s": 4}
    if name == "diffusion_X0X1X2_highpass_coverage":
        return {
            "feature_mode": "diffusion",
            "diffusion_steps": (1, 2),
            "include_highpass": True,
            "skeleton_policy": "coverage",
            "skeleton_coverage": 0.65,
            "skeleton_k_max": 8,
        }
    if name == "diffusion_highpass_coverage_adaptive":
        config = _variant_config("diffusion_X0X1X2_highpass_coverage")
        config.update({"shadow_policy": "rank_adaptive", "adaptive_b": True, "b_max": 4})
        return config
    raise ValueError(f"unknown variant: {name}")


def _mean_relation(summary: dict, key: str):
    values = [
        float(diag[key])
        for diag in summary.get("diagnostics", {}).values()
        if isinstance(diag, dict) and key in diag
    ]
    return "" if not values else sum(values) / len(values)


def _row(path: Path, dataset: str, variant: str, summary: dict) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": summary.get("seed", ""),
        "ratio": summary.get("ratio", ""),
        "loss_type": summary.get("loss_type", summary.get("ablation", {}).get("loss_type", "")),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "skeleton_coverage": _mean_relation(summary, "SkeletonMassCoverage"),
        "mean_adaptive_k": _mean_relation(summary, "mean_adaptive_k"),
        "residual_energy": _mean_relation(summary, "ResidualEnergy"),
        "shadow_recon_err": _mean_relation(summary, "ShadowReconErr"),
        "condensation_time": summary.get("condensation_time", ""),
        "training_time": summary.get("training_time", ""),
        "inference_time": summary.get("inference_time", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "status": summary.get("status", "completed"),
        "source_log": str(path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.4f}"


def _write_report(path: Path, rows: list[dict], csv_path: Path) -> None:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("accuracy") not in ("", None)]
    lines = [
        "# Medium Diffusion R+ Summary",
        "",
        "## Scope",
        "",
        "- Datasets: ogbn-arxiv, ogbn-products.",
        "- Seed: 42 only.",
        "- Ratios: 0.5%, 2.0%, 6.0%, 12.0%.",
        "- Main comparison: base vs diffusion_X0X1X2_highpass_coverage.",
        "- Ratio 6.0% includes diffusion depth, adaptive shadow, and logit-adjustment ablations.",
        "",
        "## Results",
        "",
        "| Dataset | Variant | Loss | Ratio | Acc | Macro-F1 | Pred classes | Skel cov | Recon err |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['loss_type']} | {_fmt(float(row['ratio']) * 100.0)}% | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['macro_f1'])} | {row['predicted_class_count']} | "
            f"{_fmt(row['skeleton_coverage'])} | {_fmt(row['shadow_recon_err'])} |"
        )
    for dataset in sorted({row["dataset"] for row in completed}):
        best = max([row for row in completed if row["dataset"] == dataset], key=lambda row: float(row["accuracy"]))
        lines.append(
            f"- {dataset} best: `{_fmt(best['accuracy'])}` from `{best['variant']}` / `{best['loss_type']}` at `{_fmt(float(best['ratio']) * 100.0)}%`."
        )
    failed = [row for row in rows if row.get("status") not in ("", "completed")]
    if failed:
        lines.extend(
            [
                "",
                "## OOM / OOT / Guard Failures",
                "",
                "| Dataset | Variant | Loss | Ratio | Status | Log |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for row in failed:
            ratio = "" if row.get("ratio") in ("", None) else f"{float(row['ratio']) * 100.0:.1f}%"
            lines.append(
                f"| {row['dataset']} | {row['variant']} | {row['loss_type']} | {ratio} | {row['status']} | `{row['source_log']}` |"
            )
    lines.extend(["", "## Files", "", f"- CSV: `{csv_path}`", f"- Report: `{path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medium Shadow-HGC-R+ diffusion rescue grid with seed 42.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/medium_diffusion_rplus_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/medium_diffusion_rplus_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/medium_diffusion_rplus_seed42")
    args = parser.parse_args()

    ratios = [0.005, 0.02, 0.06, 0.12]
    rows = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            path = log_dir / f"{dataset}_load_failed.json"
            payload = {"dataset": dataset, "seed": args.seed, "status": "data_not_available", "reason": str(exc)}
            write_json_summary(path, payload)
            rows.append(_row(path, dataset, "load_failed", payload))
            continue
        specs = [("base", ratio, "sqrt_weighted") for ratio in ratios]
        specs.extend(("diffusion_X0X1X2_highpass_coverage", ratio, "sqrt_weighted") for ratio in ratios)
        specs.extend(
            [
                ("diffusion_X0X1", 0.06, "sqrt_weighted"),
                ("diffusion_X0X1X2", 0.06, "sqrt_weighted"),
                ("diffusion_highpass_coverage_adaptive", 0.06, "sqrt_weighted"),
                ("diffusion_X0X1X2_highpass_coverage", 0.06, "sqrt_weighted_logit_adjusted"),
            ]
        )
        for variant, ratio, loss in specs:
            path = log_dir / f"{dataset}_{variant}_{loss}_{ratio_slug(ratio)}_seed{args.seed}.json"
            try:
                if args.skip_existing and path.exists():
                    summary = json.loads(path.read_text(encoding="utf-8"))
                else:
                    summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=path,
                        method_name="Shadow-HGC-R+" if variant != "base" else "Shadow-HGC-R-1",
                        seed=args.seed,
                        epochs=args.epochs,
                        budget_mode="ratio",
                        ratio=ratio,
                        ratio_base="train_target",
                        feature_dim=128,
                        projection_type="random",
                        loss_type=loss,
                        model_type="relation_linear",
                        min_proto_per_class=4,
                        budget_alpha=0.5,
                        multiscale_dim=128,
                        **_variant_config(variant),
                    )
                rows.append(_row(path, dataset, variant, summary))
            except Exception as exc:
                payload = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": args.seed,
                    "ratio": ratio,
                    "loss_type": loss,
                    "status": exception_status(exc),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_summary(path, payload)
                rows.append(_row(path, dataset, variant, payload))
    output = Path(args.output)
    report = Path(args.report_output)
    _write_csv(output, rows)
    _write_report(report, rows, output)
    print(f"wrote {output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
