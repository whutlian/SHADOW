from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


SMALL_DATASETS = {"acm", "dblp", "imdb"}
MEDIUM_DATASETS = {"ogbn-arxiv", "ogbn-products"}


def _load_graph(dataset: str, *, download: bool):
    if dataset in SMALL_DATASETS:
        return load_processed_small_dataset(dataset)
    return load_ogb_node_property_dataset(dataset, download=download)


def _run_defaults(dataset: str) -> dict:
    if dataset in SMALL_DATASETS:
        return {
            "feature_dim": 64,
            "projection_type": "raw",
            "loss_type": "clipped",
            "model_type": "relation_linear",
            "k_s": 2,
            "min_proto_per_class": 4,
            "budget_alpha": 0.5,
        }
    return {
        "feature_dim": 128,
        "projection_type": "random",
        "loss_type": "sqrt_weighted",
        "model_type": "relation_linear",
        "k_s": 4,
        "min_proto_per_class": 4,
        "budget_alpha": 0.5,
    }


def _rows_from_summary(path: Path, summary: dict) -> list[dict]:
    rows = []
    rank = summary.get("diagnostics", {}).get("rank", {})
    for relation, rank_diag in rank.items():
        rel_diag = summary.get("diagnostics", {}).get(relation, {})
        rows.append(
            {
                "dataset": summary.get("dataset", ""),
                "seed": summary.get("seed", ""),
                "ratio": summary.get("ratio", ""),
                "relation": relation,
                "accuracy": summary.get("accuracy", ""),
                "macro_f1": summary.get("macro_f1", ""),
                "stable_rank": rank_diag.get("stable_rank", ""),
                "entropy_effective_rank": rank_diag.get("entropy_effective_rank", ""),
                "shadow_recon_err": rel_diag.get("ShadowReconErr", ""),
                "relation_demand_norm_mean": rank_diag.get("relation_demand_norm_mean", ""),
                "relation_demand_norm_median": rank_diag.get("relation_demand_norm_median", ""),
                "relation_demand_norm_q95": rank_diag.get("relation_demand_norm_q95", ""),
                "relation_demand_norm_q995": rank_diag.get("relation_demand_norm_q995", ""),
                "shadow_feature_norm_mean": rel_diag.get("shadow_feature_norm_mean", ""),
                "shadow_feature_norm_q995": rel_diag.get("shadow_feature_norm_q995", ""),
                "status": summary.get("status", "completed"),
                "source_log": str(path),
            }
        )
    if not rows:
        rows.append(
            {
                "dataset": summary.get("dataset", ""),
                "seed": summary.get("seed", ""),
                "ratio": summary.get("ratio", ""),
                "relation": "",
                "status": summary.get("status", ""),
                "source_log": str(path),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset",
        "seed",
        "ratio",
        "relation",
        "accuracy",
        "macro_f1",
        "stable_rank",
        "entropy_effective_rank",
        "shadow_recon_err",
        "relation_demand_norm_mean",
        "relation_demand_norm_median",
        "relation_demand_norm_q95",
        "relation_demand_norm_q995",
        "shadow_feature_norm_mean",
        "shadow_feature_norm_q995",
        "status",
        "source_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _fmt(value) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.4f}"


def _write_report(path: Path, rows: list[dict], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row.get("status") == "completed" and row.get("stable_rank") not in ("", None)]
    lines = [
        "# Rank Diagnostics Summary",
        "",
        "## Scope",
        "",
        "- Seed: 42 only.",
        "- Small ratios: 0.5%, 2.5%, 9.6%.",
        "- Medium ratios: 0.5%, 6.0%, 12.0%.",
        "- Diagnostics are computed from train-target relation demand/residual matrices only.",
        "",
        "## Relation Diagnostics",
        "",
        "| Dataset | Ratio | Relation | Stable rank | Entropy rank | Recon err | Acc | Macro-F1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        lines.append(
            f"| {row['dataset']} | {_fmt(float(row['ratio']) * 100.0)}% | {row['relation']} | "
            f"{_fmt(row['stable_rank'])} | {_fmt(row['entropy_effective_rank'])} | {_fmt(row['shadow_recon_err'])} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis Checks",
            "",
            "- DBLP flatness is supported when effective ranks and reconstruction errors remain low across ratios.",
            "- IMDB failure is supported when non-target relations show high effective rank and high reconstruction error.",
            "- ACM ratio sensitivity is supported when reconstruction quality and rank leave room for target-budget gains.",
            "- Medium gaps are supported when one-hop reconstruction is moderate but accuracy remains below full-graph sanity levels, motivating diffusion features.",
            "",
            "## Files",
            "",
            f"- CSV: `{csv_path}`",
            f"- Report: `{path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shadow-HGC-R+ rank diagnostics with seed 42.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/rank_diagnostics_small_medium_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/rank_diagnostics_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/rplus_rank_diagnostics_seed42")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in args.datasets:
        ratios = [0.005, 0.025, 0.096] if dataset in SMALL_DATASETS else [0.005, 0.06, 0.12]
        try:
            graph = _load_graph(dataset, download=args.download)
        except Exception as exc:
            path = log_dir / f"{dataset}_load_failed.json"
            payload = {"dataset": dataset, "seed": args.seed, "status": "data_not_available", "reason": str(exc)}
            write_json_summary(path, payload)
            rows.extend(_rows_from_summary(path, payload))
            continue
        for ratio in ratios:
            path = log_dir / f"{dataset}_{ratio_slug(ratio)}_seed{args.seed}.json"
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
                        ratio_base="train_target",
                        **_run_defaults(dataset),
                    )
                rows.extend(_rows_from_summary(path, summary))
            except Exception as exc:
                payload = {
                    "dataset": dataset,
                    "seed": args.seed,
                    "ratio": ratio,
                    "status": exception_status(exc),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_summary(path, payload)
                rows.extend(_rows_from_summary(path, payload))
    output = Path(args.output)
    report = Path(args.report_output)
    _write_csv(output, rows)
    _write_report(report, rows, output)
    print(f"wrote {output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
