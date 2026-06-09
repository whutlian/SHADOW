from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import write_csv
from shadow_hgc.audit.parity import FULLGRAPH_PARITY_REQUIRED_FIELDS, validate_fullgraph_parity_row
from shadow_hgc.data.schema_audit import feature_hash, label_hash, schema_hash, split_hash
from shadow_hgc.data.small import load_processed_small_dataset, load_processed_small_dataset_full_schema
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.train.sehgnn_lite_target import build_schema_default_blocks, train_fullgraph_sehgnn_lite


GATES = {"acm": 0.90, "dblp": 0.90, "imdb": 0.55, "ogbn-arxiv": 0.68, "ogbn-products": 0.70}
DESIRABLE_GATES = {"acm": 0.92, "dblp": 0.91, "imdb": 0.60, "ogbn-arxiv": 0.70, "ogbn-products": 0.72}


def _num_classes(labels) -> int:
    valid = labels[labels >= 0]
    return 0 if valid.numel() == 0 else int(valid.max().item()) + 1


def _class_count(labels, idx) -> int:
    if idx.numel() == 0:
        return 0
    selected = labels[idx]
    return int(selected[selected >= 0].unique().numel())


def _base_fields(graph, *, dataset: str, variant: str, seed: int, summary: dict, status: str, reason: str = "") -> dict:
    acc = summary.get("accuracy", "")
    gate = GATES[dataset]
    gate_passed = acc != "" and float(acc) >= gate
    row = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "target_type": graph.target_type,
        "status": status,
        "reason": reason or summary.get("reason") or "completed",
        "accuracy": acc,
        "macro_f1": summary.get("macro_f1", ""),
        "weighted_f1": summary.get("weighted_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", ""),
        "target_gate": gate,
        "desirable_gate": DESIRABLE_GATES[dataset],
        "gate_passed": gate_passed,
        "blocked_by_fullgraph_backbone": not gate_passed,
        "model_type": summary.get("model_type", "sehgnn_lite"),
        "feature_mode": summary.get("feature_mode", "schema_default_metapath"),
        "metapath_blocks": json.dumps(summary.get("metapath_blocks", [])),
        "path_lad_blocks": json.dumps(summary.get("path_lad_blocks", [])),
        "num_nodes_by_type": json.dumps({key: int(value) for key, value in sorted(graph.num_nodes.items())}),
        "num_edges_by_type": json.dumps({str(rel): int(graph.edge_index[rel].shape[1]) for rel in graph.relations}, sort_keys=True),
        "split_hash": split_hash(graph),
        "feature_hash": feature_hash(graph),
        "label_hash": label_hash(graph),
        "schema_hash": schema_hash(graph),
        "train_nodes": int(graph.train_idx.numel()),
        "valid_nodes": int(graph.val_idx.numel()),
        "test_nodes": int(graph.test_idx.numel()),
        "num_classes": _num_classes(graph.labels),
        "train_class_count": _class_count(graph.labels, graph.train_idx),
        "valid_class_count": _class_count(graph.labels, graph.val_idx),
        "test_class_count": _class_count(graph.labels, graph.test_idx),
        "training_time_s": summary.get("train_time", summary.get("training_time_s", "")),
        "inference_time_s": summary.get("infer_time", summary.get("inference_time_s", "")),
        "peak_cpu_ram_mb": current_cpu_ram_bytes() / (1024**2),
        "peak_gpu_ram_mb": current_gpu_ram_bytes() / (1024**2),
        "schema_incomplete_for_sota": summary.get("schema_incomplete_for_sota", False),
        "source_log": summary.get("source_log", ""),
    }
    checks = validate_fullgraph_parity_row(row)
    row["invalid_reasons"] = json.dumps(checks["reasons"])
    return row


def _run_small(dataset: str, variant: str, args, *, full_schema: bool, hidden_dim: int, dropout: float, lr: float) -> dict:
    graph = load_processed_small_dataset_full_schema(dataset) if full_schema else load_processed_small_dataset(dataset)
    log_path = Path(args.log_dir) / f"{dataset}_{variant}_seed{args.seed}.json"
    if args.skip_existing and log_path.exists():
        summary = json.loads(log_path.read_text(encoding="utf-8"))
        return _base_fields(graph, dataset=dataset, variant=variant, seed=args.seed, summary={**summary, "source_log": str(log_path)}, status=summary.get("status", "completed"), reason=summary.get("reason", ""))
    try:
        blocks, metadata = build_schema_default_blocks(graph, include_self=True, include_metapath=True)
        run = train_fullgraph_sehgnn_lite(
            graph,
            blocks=blocks,
            metadata=metadata,
            seed=args.seed,
            epochs=args.epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=5e-4 if "tuned" in variant else 1e-4,
            loss_type="weighted",
        )
        summary = {"dataset": dataset, "variant": variant, "seed": args.seed, "status": "completed", **run.summary}
        if dataset == "dblp" and "APA" in summary.get("metapath_blocks", []) and len(summary.get("metapath_blocks", [])) <= 1:
            summary["schema_incomplete_for_sota"] = True
        write_json_summary(log_path, summary)
    except Exception as exc:
        summary = {"dataset": dataset, "variant": variant, "seed": args.seed, "status": "experiment_failed", "reason": str(exc)}
        write_json_summary(log_path, summary)
    return _base_fields(graph, dataset=dataset, variant=variant, seed=args.seed, summary={**summary, "source_log": str(log_path)}, status=summary.get("status", "completed"), reason=summary.get("reason", ""))


def _skipped_small(dataset: str, variant: str, args, *, full_schema: bool, reason: str) -> dict:
    graph = load_processed_small_dataset_full_schema(dataset) if full_schema else load_processed_small_dataset(dataset)
    summary = {
        "dataset": dataset,
        "variant": variant,
        "seed": args.seed,
        "status": "skipped_optional_not_implemented",
        "reason": reason,
        "model_type": "not_run",
        "feature_mode": "optional_backbone_not_run",
    }
    return _base_fields(graph, dataset=dataset, variant=variant, seed=args.seed, summary=summary, status="skipped_optional_not_implemented", reason=reason)


def _medium_rows(seed: int) -> list[dict]:
    rows = []
    for dataset in ["ogbn-arxiv", "ogbn-products"]:
        log_path = Path("experiments/logs/lad_stage_diagnostics_seed42") / f"{dataset}_FullDemandTable-MLP_r0p12_seed42.json"
        summary = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
        acc = summary.get("accuracy", "")
        gate = GATES[dataset]
        gate_passed = acc != "" and float(acc) >= gate
        base = {
            "dataset": dataset,
            "variant": "fullgraph_lad_table_teacher",
            "seed": seed,
            "target_type": "paper" if dataset == "ogbn-arxiv" else "product",
            "status": "completed_existing_diagnostic" if summary else "missing_existing_diagnostic",
            "reason": "completed_existing_diagnostic" if summary else "FullDemandTable-MLP log missing",
            "accuracy": acc,
            "macro_f1": summary.get("macro_f1", ""),
            "weighted_f1": summary.get("weighted_f1", ""),
            "predicted_class_count": summary.get("predicted_class_count", ""),
            "target_gate": gate,
            "desirable_gate": DESIRABLE_GATES[dataset],
            "gate_passed": gate_passed,
            "blocked_by_fullgraph_backbone": not gate_passed,
            "model_type": "full_demand_table_mlp",
            "feature_mode": "no_diffusion_lad_table",
            "metapath_blocks": "[]",
            "path_lad_blocks": "[]",
            "num_nodes_by_type": "{}",
            "num_edges_by_type": "{}",
            "split_hash": summary.get("split_hash", "existing_diagnostic_no_hash"),
            "feature_hash": summary.get("feature_hash", "existing_diagnostic_no_hash"),
            "label_hash": summary.get("label_hash", "existing_diagnostic_no_hash"),
            "schema_hash": summary.get("schema_hash", "homogeneous"),
            "train_nodes": summary.get("train_nodes", "not_logged_existing_diagnostic"),
            "valid_nodes": summary.get("valid_nodes", "not_logged_existing_diagnostic"),
            "test_nodes": summary.get("test_nodes", "not_logged_existing_diagnostic"),
            "num_classes": summary.get("num_classes", "not_logged_existing_diagnostic"),
            "train_class_count": summary.get("train_class_count", "not_logged_existing_diagnostic"),
            "valid_class_count": summary.get("valid_class_count", "not_logged_existing_diagnostic"),
            "test_class_count": summary.get("test_class_count", "not_logged_existing_diagnostic"),
            "training_time_s": summary.get("train_time_s", ""),
            "inference_time_s": summary.get("inference_time_s", ""),
            "peak_cpu_ram_mb": summary.get("peak_cpu_ram_mb", summary.get("peak_cpu_ram", "")),
            "peak_gpu_ram_mb": summary.get("peak_gpu_ram_mb", summary.get("peak_gpu_ram", "")),
            "source_log": str(log_path),
            "invalid_reasons": "[]",
        }
        rows.append(base)
        if dataset == "ogbn-products":
            for variant in ["fullgraph_lad_table_teacher_balanced_softmax", "fullgraph_lad_table_teacher_logit_adjusted"]:
                row = dict(base)
                row.update({"variant": variant, "status": "skipped_resource_guard", "reason": "products calibration teacher rows are guarded; no P2/diffusion path is run", "accuracy": "", "macro_f1": "", "weighted_f1": "", "gate_passed": False, "blocked_by_fullgraph_backbone": True})
                rows.append(row)
    return rows


def _write_report(rows: list[dict], path: Path, csv_path: Path) -> None:
    lines = ["# Fullgraph Parity Seed 42", "", "| Dataset | Variant | Acc | Gate | Passed | Blocked | Status | Reason |", "|---|---|---:|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row.get('accuracy','')} | {row.get('target_gate','')} | {row.get('gate_passed','')} | {row.get('blocked_by_fullgraph_backbone','')} | {row.get('status','')} | {row.get('reason','')} |")
    lines.extend(["", "Rows below gate set `blocked_by_fullgraph_backbone=true`; downstream promoted condensation is blocked for those datasets.", "", f"- CSV: `{csv_path}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fullgraph parity audit, seed 42 only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/fullgraph_parity_seed42")
    parser.add_argument("--output", default="experiments/tables/fullgraph_parity_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/fullgraph_parity_seed42.md")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(_run_small("acm", "fullgraph_sehgnn_lite_current", args, full_schema=False, hidden_dim=128, dropout=0.3, lr=0.01))
    rows.append(_run_small("acm", "fullgraph_sehgnn_lite_tuned", args, full_schema=False, hidden_dim=256, dropout=0.2, lr=0.003))
    rows.append(_skipped_small("acm", "fullgraph_han_style_optional", args, full_schema=False, reason="HAN-style optional backbone is not implemented in this sprint; row is skipped and excluded from best summaries"))
    rows.append(_run_small("dblp", "fullgraph_dblp_APA_only", args, full_schema=False, hidden_dim=128, dropout=0.3, lr=0.01))
    rows.append(_run_small("dblp", "fullgraph_dblp_full_schema_sehgnn_lite", args, full_schema=True, hidden_dim=256, dropout=0.3, lr=0.003))
    rows.append(_run_small("imdb", "fullgraph_imdb_sehgnn_lite_MAM_MDM_MKM", args, full_schema=False, hidden_dim=128, dropout=0.3, lr=0.01))
    rows.append(_skipped_small("imdb", "fullgraph_imdb_han_style_optional", args, full_schema=False, reason="HAN-style optional backbone is not implemented in this sprint; row is skipped and excluded from best summaries"))
    rows.extend(_medium_rows(args.seed))
    output = Path(args.output)
    write_csv(output, rows, [*FULLGRAPH_PARITY_REQUIRED_FIELDS, "desirable_gate", "schema_incomplete_for_sota", "source_log", "invalid_reasons"])
    _write_report(rows, Path(args.report), output)


if __name__ == "__main__":
    main()
