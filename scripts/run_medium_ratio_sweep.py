from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
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


def _parse_ratios(args: argparse.Namespace) -> list[float]:
    if args.ratios:
        return [float(value) for value in args.ratios]
    values = []
    current = int(round(args.ratio_start_percent * 10))
    stop = int(round(args.ratio_stop_percent * 10))
    step = int(round(args.ratio_step_percent * 10))
    if step <= 0:
        raise ValueError("--ratio-step-percent must be positive")
    while current <= stop:
        values.append(current / 1000.0)
        current += step
    return values


def _row_from_summary(path: Path, summary: dict) -> dict:
    diagnostics = summary.get("diagnostics") or {}
    diag_values = [
        diag
        for diag in diagnostics.values()
        if isinstance(diag, dict) and "ShadowReconErr" in diag
    ]
    if diag_values:
        skeleton = sum(float(d.get("SkeletonMassCoverage", 0.0)) for d in diag_values) / len(diag_values)
        residual = sum(float(d.get("ResidualEnergy", 0.0)) for d in diag_values) / len(diag_values)
        recon = sum(float(d.get("ShadowReconErr", 0.0)) for d in diag_values) / len(diag_values)
    else:
        skeleton = residual = recon = ""
    return {
        "dataset": summary.get("dataset", ""),
        "seed": summary.get("seed", ""),
        "ratio": summary.get("ratio", ""),
        "ratio_percent": "" if summary.get("ratio") in (None, "") else float(summary["ratio"]) * 100.0,
        "status": summary.get("status", "completed"),
        "method": summary.get("method", ""),
        "loss_type": summary.get("loss_type", ""),
        "model": summary.get("model", ""),
        "projection_type": summary.get("projection_type", ""),
        "requested_target_budget": summary.get("requested_target_budget", ""),
        "effective_target_prototypes": summary.get("effective_target_prototypes", ""),
        "shadow_nodes_total": summary.get("shadow_nodes_total", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "effective_target_ratio": summary.get("effective_target_ratio", ""),
        "condensed_node_ratio_to_train_target": summary.get("condensed_node_ratio_to_train_target", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_classes": summary.get("predicted_classes", ""),
        "skeleton_coverage_mean": skeleton,
        "residual_energy_mean": residual,
        "shadow_recon_err_mean": recon,
        "condense_s": summary.get("condensation_time", ""),
        "train_s": summary.get("training_time", ""),
        "infer_s": summary.get("inference_time", ""),
        "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", ""),
        "source_log": str(path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "seed",
        "ratio",
        "ratio_percent",
        "status",
        "method",
        "loss_type",
        "model",
        "projection_type",
        "requested_target_budget",
        "effective_target_prototypes",
        "shadow_nodes_total",
        "condensed_nodes_total",
        "condensed_edges_total",
        "effective_target_ratio",
        "condensed_node_ratio_to_train_target",
        "accuracy",
        "macro_f1",
        "predicted_classes",
        "skeleton_coverage_mean",
        "residual_energy_mean",
        "shadow_recon_err_mean",
        "condense_s",
        "train_s",
        "infer_s",
        "peak_cpu_ram_gb",
        "peak_gpu_ram_gb",
        "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _format_float(value: object, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def _write_md(path: Path, rows: list[dict], csv_path: Path, log_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Medium Ratio Sweep Summary",
        "",
        "## Scope",
        "",
        "- Datasets: ogbn-arxiv, ogbn-products.",
        "- Seed: 42.",
        "- Ratios: 0.5% to 12.0%, 0.5 percentage-point spacing unless overridden.",
        "- Setting: Shadow-HGC-R-1, relation-linear, sqrt-weighted prototype loss, random projection, 500 epochs.",
        "- Ratio is requested target prototype ratio; condensed node ratio is higher because shadow nodes are added.",
        "",
        "## Best Points",
        "",
        "| Dataset | Best ratio by accuracy | Accuracy | Macro-F1 | Condensed nodes | Predicted classes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({str(row["dataset"]) for row in rows if row.get("status", "completed") == "completed"}):
        completed = [row for row in rows if row.get("dataset") == dataset and row.get("status", "completed") == "completed" and row.get("accuracy") not in ("", None)]
        if not completed:
            continue
        best = max(completed, key=lambda row: float(row["accuracy"]))
        lines.append(
            f"| {dataset} | {_format_float(best['ratio_percent'], 1)}% | {_format_float(best['accuracy'])} | "
            f"{_format_float(best['macro_f1'])} | {best.get('condensed_nodes_total', '')} | {best.get('predicted_classes', '')} |"
        )
    lines.extend(["", "## Accuracy Curve", ""])
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = sorted([row for row in rows if row.get("dataset") == dataset], key=lambda row: float(row.get("ratio") or 0.0))
        pairs = []
        for row in dataset_rows:
            if row.get("accuracy") in ("", None):
                pairs.append(f"{_format_float(row.get('ratio_percent'), 1)}%=NA({row.get('status', '')})")
            else:
                pairs.append(f"{_format_float(row.get('ratio_percent'), 1)}%={_format_float(row.get('accuracy'))}")
        lines.append(f"- {dataset}: " + "; ".join(pairs))
    lines.extend(["", "## Files", "", f"- CSV: `{csv_path}`", f"- Logs: `{log_dir}`", f"- Summary: `{path}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a medium-dataset ratio sweep without ablations.")
    parser.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ratios", nargs="*", type=float)
    parser.add_argument("--ratio-start-percent", type=float, default=0.5)
    parser.add_argument("--ratio-stop-percent", type=float, default=12.0)
    parser.add_argument("--ratio-step-percent", type=float, default=0.5)
    parser.add_argument("--ratio-base", choices=["train_target", "all_target"], default="train_target")
    parser.add_argument("--output", default="experiments/tables/medium_ratio_sweep_seed42_20260608.csv")
    parser.add_argument("--summary-output", default="experiments/reports/medium_ratio_sweep_seed42_20260608.md")
    parser.add_argument("--log-dir", default="experiments/logs/medium_ratio_sweep_seed42_20260608")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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
    parser.add_argument("--model", choices=["relation_linear", "relation_mlp"], default="relation_linear")
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
    parser.add_argument("--shadow-ratio-target-target", type=float, default=0.5)
    parser.add_argument("--shadow-ratio-non-target", type=float, default=1.0)
    parser.add_argument("--min-shadow-per-relation", type=int, default=8)
    args = parser.parse_args()

    ratios = _parse_ratios(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()

    for dataset in args.datasets:
        print(f"[dataset] loading {dataset}", flush=True)
        try:
            graph = load_ogb_node_property_dataset(dataset, download=args.download)
        except Exception as exc:
            path = log_dir / f"{dataset}_load_failed_seed{args.seed}.json"
            payload = {
                "dataset": dataset,
                "seed": args.seed,
                "method": "Shadow-HGC-R-1",
                "status": "ogb_data_not_available",
                "reason": str(exc),
            }
            write_json_summary(path, payload)
            rows.append(_row_from_summary(path, payload))
            continue

        for ratio in ratios:
            label = ratio_slug(ratio)
            path = log_dir / f"{dataset}_{label}_seed{args.seed}.json"
            print(f"[run] {dataset} seed={args.seed} ratio={ratio:.4f}", flush=True)
            try:
                if args.skip_existing and path.exists():
                    summary = json.loads(path.read_text(encoding="utf-8"))
                else:
                    summary = run_shadow_hgc_experiment(
                        graph,
                        output_path=path,
                        seed=args.seed,
                        epochs=args.epochs,
                        budget_mode="ratio",
                        ratio=ratio,
                        ratio_base=args.ratio_base,
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
                rows.append(_row_from_summary(path, summary))
                print(
                    f"[done] {dataset} ratio={ratio:.4f} acc={summary.get('accuracy')} "
                    f"f1={summary.get('macro_f1')} nodes={summary.get('condensed_nodes_total')}",
                    flush=True,
                )
            except Exception as exc:
                status = exception_status(exc)
                payload = {
                    "dataset": dataset,
                    "seed": args.seed,
                    "method": "Shadow-HGC-R-1",
                    "status": status,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                    "budget_mode": "ratio",
                    "ratio": ratio,
                    "ratio_base": args.ratio_base,
                    "loss_type": args.loss_type,
                    "model": args.model,
                    "projection_type": args.projection_type,
                }
                write_json_summary(path, payload)
                rows.append(_row_from_summary(path, payload))
                print(f"[error] {dataset} ratio={ratio:.4f} status={status} reason={exc}", flush=True)

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    rows.sort(key=lambda row: (str(row.get("dataset", "")), float(row.get("ratio") or 0.0)))
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    _write_csv(output, rows)
    _write_md(summary_output, rows, output, log_dir)
    print(f"[summary] wrote {output}", flush=True)
    print(f"[summary] wrote {summary_output}", flush=True)
    print(f"[summary] elapsed_s={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
