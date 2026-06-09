from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.train.sehgnn_lite_target import build_schema_default_blocks, train_fullgraph_sehgnn_lite


GATES = {
    "acm": 0.90,
    "dblp": 0.88,
    "imdb": 0.55,
    "ogbn-arxiv": 0.65,
    "ogbn-products": 0.70,
}


def _small_row(dataset: str, args) -> dict:
    log_path = Path(args.log_dir) / f"{dataset}_fullgraph_sehgnn_lite_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        summary = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        graph = load_processed_small_dataset(dataset)
        blocks, metadata = build_schema_default_blocks(
            graph,
            include_self=True,
            include_metapath=True,
            include_path_lad_v2=False,
        )
        run = train_fullgraph_sehgnn_lite(
            graph,
            blocks=blocks,
            metadata=metadata,
            seed=args.seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            dropout=0.3,
            lr=0.01,
            weight_decay=1e-4,
            loss_type="weighted",
        )
        summary = {
            "dataset": dataset,
            "variant": "fullgraph_sehgnn_lite",
            "seed": args.seed,
            "status": "completed",
            "target_type": graph.target_type,
            "use_diffusion": False,
            "blocked_by_fullgraph_backbone": run.summary["accuracy"] < GATES[dataset],
            **run.summary,
        }
        write_json_summary(log_path, summary)
    return {
        "dataset": dataset,
        "variant": "fullgraph_sehgnn_lite",
        "seed": summary.get("seed", args.seed),
        "status": summary.get("status", "completed"),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", ""),
        "target_gate": GATES[dataset],
        "gate_passed": summary.get("accuracy", 0.0) >= GATES[dataset] if summary.get("accuracy", "") != "" else False,
        "blocked_by_fullgraph_backbone": summary.get("blocked_by_fullgraph_backbone", ""),
        "model_type": summary.get("model_type", ""),
        "metapath_blocks": json.dumps(summary.get("metapath_blocks", [])),
        "block_norm_stats_source": summary.get("block_norm_stats_source", ""),
        "use_diffusion": summary.get("use_diffusion", False),
        "source_log": str(log_path),
    }


def _medium_existing_row(dataset: str, args) -> dict:
    name = f"{dataset}_FullDemandTable-MLP_r0p12_seed42.json"
    log_path = Path("experiments/logs/lad_stage_diagnostics_seed42") / name
    summary = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    status = "completed_existing_diagnostic" if summary else "skipped_missing_existing_full_demand_table"
    accuracy = summary.get("accuracy", "")
    return {
        "dataset": dataset,
        "variant": "fullgraph_no_diffusion_lad_table_teacher",
        "seed": args.seed,
        "status": status,
        "accuracy": accuracy,
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", ""),
        "target_gate": GATES[dataset],
        "gate_passed": float(accuracy) >= GATES[dataset] if accuracy != "" else False,
        "blocked_by_fullgraph_backbone": (float(accuracy) < GATES[dataset]) if accuracy != "" else True,
        "model_type": summary.get("model_type", "full_demand_table_mlp"),
        "metapath_blocks": "[]",
        "lad_blocks": json.dumps(["P1", "P2"]),
        "block_norm_stats_source": summary.get("compiled_block_stats_source", "train_full_demand_table"),
        "use_diffusion": False,
        "source_log": str(log_path),
        "reason": "" if summary else "existing full demand table diagnostic log not found",
    }


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = [
        "# Fullgraph Backbone Audit Seed 42",
        "",
        "Small datasets use full-target-row SeHGNNLite with schema-default target feature blocks. Medium rows use the existing no-diffusion FullDemandTable-MLP diagnostics as the table-teacher audit.",
        "",
        "| Dataset | Variant | Acc | Macro-F1 | Gate | Passed | Blocked | Status |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row.get('accuracy','')} | {row.get('macro_f1','')} | "
            f"{row.get('target_gate','')} | {row.get('gate_passed','')} | {row.get('blocked_by_fullgraph_backbone','')} | {row.get('status','')} |"
        )
    lines.extend(["", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fullgraph/backbone audit for SOTA clean sprint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/fullgraph_backbone_audit_seed42")
    parser.add_argument("--output", default="experiments/tables/fullgraph_backbone_audit_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/fullgraph_backbone_audit_seed42.md")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = [_small_row(dataset, args) for dataset in ["acm", "dblp", "imdb"]]
    rows.extend(_medium_existing_row(dataset, args) for dataset in ["ogbn-arxiv", "ogbn-products"])
    output = Path(args.output)
    write_csv(output, rows)
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()

