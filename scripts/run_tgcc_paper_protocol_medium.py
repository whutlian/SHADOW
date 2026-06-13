from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_full_node_ratio import _budget_for_full_node_ratio
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


DATASETS = ["ogbn-arxiv", "ogbn-products"]
PAPER_FULL_NODE_RATIOS = [0.0005, 0.0025, 0.005]


def _ratio_label(ratio: float) -> str:
    return str(float(ratio)).replace(".", "p")


def _run_one(graph, dataset: str, ratio: float, seed: int, args) -> dict:
    target_budget, shadow_budget, budget_meta = _budget_for_full_node_ratio(
        graph,
        ratio,
        min_proto_per_class=1,
    )
    log_path = Path(args.log_dir) / f"{dataset}_tgcc_protocol_fullnode_r{_ratio_label(ratio)}_seed{seed}.json"
    if args.skip_existing and log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    try:
        summary = run_shadow_hgc_experiment(
            graph,
            output_path=log_path,
            method_name="Shadow-HGC-R-1",
            stage="tgcc_paper_protocol_full_node_ratio",
            seed=seed,
            epochs=args.epochs,
            budget_mode="count",
            target_budget=target_budget,
            feature_dim=args.feature_dim,
            projection_type="random",
            degree_scale=0.1,
            loss_type=args.loss_type,
            model_type="relation_linear",
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            min_proto_per_class=1,
            budget_alpha=0.5,
            k_s=args.k_s,
            shadow_policy="fixed",
            M_r=shadow_budget,
            assignment_chunk_size=args.chunk_size,
            inference_dst_chunk_size=args.chunk_size,
            demand_edge_chunk_size=args.chunk_size,
            inference_edge_chunk_size=args.chunk_size,
        )
        summary.update(budget_meta)
        summary["requested_full_condensed_node_ratio"] = float(ratio)
        summary["actual_full_condensed_node_ratio"] = summary.get("total_condensed_node_ratio")
        summary["actual_full_condensed_node_ratio_error"] = (
            float(summary["actual_full_condensed_node_ratio"]) - float(ratio)
            if summary.get("actual_full_condensed_node_ratio") is not None
            else None
        )
        summary["paper_reference"] = "TGCC AAAI2026 Table 1/2 protocol: r=m/N, 5 random seeds, OGB arxiv ratios 0.05/0.25/0.5%."
        summary["paper_product_note"] = (
            "ogbn-products is not listed in the TGCC paper/code; this row adapts the same OGB full-node ratio protocol."
            if dataset == "ogbn-products"
            else ""
        )
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        status = exception_status(exc)
        payload = {
            "dataset": dataset,
            "seed": seed,
            "status": status,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "requested_full_condensed_node_ratio": float(ratio),
            **budget_meta,
        }
        write_json_summary(log_path, payload)
        return payload


def _row(summary: dict, dataset: str, ratio: float, seed: int) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "requested_full_condensed_node_ratio": ratio,
        "actual_full_condensed_node_ratio": summary.get("actual_full_condensed_node_ratio", summary.get("total_condensed_node_ratio", "")),
        "actual_ratio_error": summary.get("actual_full_condensed_node_ratio_error", ""),
        "planned_total_condensed_nodes": summary.get("planned_total_condensed_nodes", ""),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "effective_M_tau": summary.get("effective_M_tau", ""),
        "shadow_nodes_total": summary.get("shadow_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("predicted_classes", "")),
        "condensation_time": summary.get("condensation_time", ""),
        "training_time": summary.get("training_time", ""),
        "inference_time": summary.get("inference_time", ""),
        "peak_cpu_ram": summary.get("peak_cpu_ram", ""),
        "peak_gpu_ram": summary.get("peak_gpu_ram", ""),
        "status": summary.get("status", "completed"),
        "reason": summary.get("reason", ""),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _float_or_none(value) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _aggregate(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(row["dataset"], float(row["requested_full_condensed_node_ratio"])) for row in rows})
    for dataset, ratio in keys:
        group = [
            row for row in rows
            if row["dataset"] == dataset
            and float(row["requested_full_condensed_node_ratio"]) == ratio
            and row.get("status", "completed") == "completed"
        ]
        acc = [value for row in group if (value := _float_or_none(row.get("accuracy"))) is not None]
        macro = [value for row in group if (value := _float_or_none(row.get("macro_f1"))) is not None]
        actual = [value for row in group if (value := _float_or_none(row.get("actual_full_condensed_node_ratio"))) is not None]
        out.append(
            {
                "dataset": dataset,
                "requested_full_condensed_node_ratio": ratio,
                "runs_completed": len(group),
                "accuracy_mean": _mean(acc),
                "accuracy_std": _std(acc),
                "macro_f1_mean": _mean(macro),
                "macro_f1_std": _std(macro),
                "actual_full_condensed_node_ratio_mean": _mean(actual),
                "status": "completed" if len(group) > 0 else "no_completed_runs",
            }
        )
    return out


def _fmt(value, *, pct: bool = False) -> str:
    if value in ("", None):
        return ""
    value = float(value)
    if pct:
        return f"{value * 100:.4f}%"
    return f"{value:.4f}"


def _write_report(path: Path, rows: list[dict], aggregate_rows: list[dict], csv_path: Path, aggregate_path: Path, log_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Shadow-HGC-R-1 Under TGCC Paper Settings",
        "",
        "## Scope",
        "",
        "- Method actually run: `Shadow-HGC-R-1` main path from this repository.",
        "- Paper source: `D:/paper/downloaded_papers/TGCC - Transferable Graph Condensation from the Causal Perspective.pdf`.",
        "- Experiment protocol borrowed from the paper: full-graph node reduction ratio `r=m/N`, OGB arxiv ratios `0.05%/0.25%/0.5%`, and 5 random seeds.",
        "- `ogbn-products` is not listed in the TGCC paper or official code; products rows adapt the same OGB full-node ratio protocol.",
        "- This is not a TGCC implementation run. The paper is used only to define the evaluation setting.",
        "- Training hyperparameters were aligned where applicable with the official TGCC OGB config: 800 epochs, hidden 256, dropout 0, lr 0.01, weight decay 0.",
        "",
        "## Paper And Code Audit",
        "",
        "- The PDF main text reports OGB arxiv in Table 1 and Table 2. It does not include ogbn-products.",
        "- The PDF states that single-task/single-dataset node-classification results and detailed hyperparameters are in appendices, but those appendix sections are not present in the provided 9-page PDF.",
        "- The official TGCC repository at `D:/paper/code/TGCC` has config entries for `ogbn-arxiv-r0.001`, `ogbn-arxiv-r0.005`, and `ogbn-arxiv-r0.01`, not for `0.0005/0.0025/0.005` exactly and not for ogbn-products.",
        "- The official TGCC runner expects GraphSAINT-style files such as `data/<dataset>/adj_full.npz`, `feats.npy`, `role.json`, `class_map.json`, plus spectral/perturbation `.npz` files. These are not available in the current project dataset cache.",
        "- Therefore, the executable experiment below is our method (`Shadow-HGC-R-1`) evaluated under the visible TGCC paper settings, not a direct TGCC reproduction.",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Requested full node ratio | Actual full node ratio mean | Runs | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['requested_full_condensed_node_ratio'], pct=True)} | "
            f"{_fmt(row['actual_full_condensed_node_ratio_mean'], pct=True)} | {row['runs_completed']} | "
            f"{_fmt(row['accuracy_mean'])} | {_fmt(row['accuracy_std'])} | "
            f"{_fmt(row['macro_f1_mean'])} | {_fmt(row['macro_f1_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Dataset | Seed | Requested full node ratio | Actual full node ratio | Condensed nodes | Target prototypes | Shadow nodes | Accuracy | Macro-F1 | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {_fmt(row['requested_full_condensed_node_ratio'], pct=True)} | "
            f"{_fmt(row['actual_full_condensed_node_ratio'], pct=True)} | {row['condensed_nodes_total']} | "
            f"{row['effective_M_tau']} | {row['shadow_nodes_total']} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['macro_f1'])} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-seed CSV: `{csv_path}`",
            f"- Aggregate CSV: `{aggregate_path}`",
            f"- JSON logs: `{log_dir}`",
            f"- Report: `{path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run arxiv/products under TGCC paper full-node-ratio protocol.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--ratios", nargs="+", type=float, default=PAPER_FULL_NODE_RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", default="sqrt_weighted")
    parser.add_argument("--k-s", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/tgcc_paper_protocol_medium")
    parser.add_argument("--output", default="experiments/tables/tgcc_paper_protocol_medium_seed0_4.csv")
    parser.add_argument("--aggregate-output", default="experiments/tables/tgcc_paper_protocol_medium_aggregate.csv")
    parser.add_argument("--report", default="experiments/reports/tgcc_paper_protocol_medium_summary.md")
    args = parser.parse_args()

    started = time.time()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in args.datasets:
        print(f"[dataset] loading {dataset}", flush=True)
        graph = load_ogb_node_property_dataset(dataset, download=args.download)
        for ratio in args.ratios:
            for seed in args.seeds:
                print(f"[run] {dataset} ratio={ratio:.6f} seed={seed}", flush=True)
                summary = _run_one(graph, dataset, ratio, seed, args)
                rows.append(_row(summary, dataset, ratio, seed))
                print(
                    f"[done] {dataset} ratio={ratio:.6f} seed={seed} "
                    f"status={summary.get('status', 'completed')} acc={summary.get('accuracy')}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["dataset"], float(row["requested_full_condensed_node_ratio"]), int(row["seed"])))
    aggregate_rows = _aggregate(rows)
    aggregate_rows.sort(key=lambda row: (row["dataset"], float(row["requested_full_condensed_node_ratio"])))
    output = Path(args.output)
    aggregate_output = Path(args.aggregate_output)
    _write_csv(output, rows)
    _write_csv(aggregate_output, aggregate_rows)
    _write_report(Path(args.report), rows, aggregate_rows, output, aggregate_output, Path(args.log_dir))
    print(f"[summary] wrote {output}", flush=True)
    print(f"[summary] wrote {aggregate_output}", flush=True)
    print(f"[summary] wrote {args.report}", flush=True)
    print(f"[summary] elapsed_s={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
