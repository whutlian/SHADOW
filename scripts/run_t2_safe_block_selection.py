from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import (
    ALL_T2_DATASETS,
    MEDIUM_DATASETS,
    SAFE_BASELINES,
    T2_STAGE_FIELDS,
    build_t2_block_groups,
    group_resource,
    load_t2_graph,
    make_stage_row,
    markdown_table,
    merge_block_groups,
    num_classes,
    promotion_status,
    split_train_valid,
    write_csv,
    write_json,
)
from shadow_hgc.train.train_sft_teacher import train_sft_teacher


def _candidate_configs(dataset: str, args) -> list[dict[str, Any]]:
    if dataset in MEDIUM_DATASETS:
        return [
            {"model_type": "sagn_lite", "hidden_dim": args.medium_hidden_dim, "dropout": 0.3, "loss_type": "cross_entropy", "lr": args.lr, "weight_decay": args.weight_decay},
            {"model_type": "gamlp_lite", "hidden_dim": args.medium_hidden_dim, "dropout": 0.3, "loss_type": "balanced_softmax", "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    return [
        {"model_type": "sagn_lite", "hidden_dim": args.hidden_dim, "dropout": 0.3, "loss_type": "cross_entropy", "lr": args.lr, "weight_decay": args.weight_decay},
        {"model_type": "gamlp_lite", "hidden_dim": args.hidden_dim, "dropout": 0.3, "loss_type": "class_balanced_ce", "lr": args.lr, "weight_decay": args.weight_decay},
    ]


def _run_training(blocks, graph, train_rows, valid_rows, args, config):
    medium = graph.dataset_name in MEDIUM_DATASETS
    return train_sft_teacher(
        blocks,
        graph.labels,
        train_rows,
        valid_rows,
        graph.test_idx,
        num_classes=num_classes(graph.labels),
        model_type=config["model_type"],
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        loss_type=config["loss_type"],
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        epochs=args.medium_epochs if medium else args.epochs,
        patience=args.patience,
        seed=args.seed,
        batch_size=args.medium_batch_size if medium else None,
        label_smoothing=args.label_smoothing if config["loss_type"] == "label_smoothing_ce" else 0.0,
    )


def _resource_guard_row(dataset: str, args, *, reason: str) -> dict[str, Any]:
    safe = SAFE_BASELINES[dataset]
    return {
        "dataset": dataset,
        "row_kind": "resource_guard",
        "model_type": "",
        "block_group": "",
        "selected_blocks": "[]",
        "status": "blocked_resource_guard",
        "reason": reason,
        "safe_baseline": safe["variant"],
        "safe_baseline_acc": safe["accuracy"],
        "safe_baseline_macro_f1": safe["macro_f1"],
        "full_edge_scans": 0,
        "edge_chunk_size": args.edge_chunk_size,
        "dst_chunk_size": args.dst_chunk_size,
        "block_dim": args.medium_block_dim,
        "num_blocks": 0,
        "cache_bytes": 0,
        "uses_memmap": True,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_logits_as_input": False,
        "uses_bounded_edges": False,
        "uses_diffusion_legacy": False,
        "uses_full_graph_backprop": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_train_labels_only": False,
        "source_log": "",
    }


def run_dataset_selection(dataset: str, args) -> list[dict[str, Any]]:
    started = time.perf_counter()
    if dataset == "ogbn-products" and not args.run_products_full:
        return [_resource_guard_row(dataset, args, reason="products full T2 SFT skipped locally; use --run-products-full after dry-run")]
    graph = load_t2_graph(dataset)
    train_rows, valid_rows = split_train_valid(graph, seed=args.seed)
    medium = dataset in MEDIUM_DATASETS
    block_dim = args.medium_block_dim if medium else args.block_dim
    groups, diagnostics = build_t2_block_groups(
        graph,
        train_rows_for_labels=train_rows,
        seed=args.seed,
        block_dim=block_dim,
        edge_chunk_size=args.edge_chunk_size,
        edge_limit=args.edge_limit,
        scap_topk=args.scap_topk,
    )
    rows: list[dict[str, Any]] = []
    configs = _candidate_configs(dataset, args)
    best_base = None
    best_config = None
    for config in configs:
        result = _run_training(groups["B0_self"], graph, train_rows, valid_rows, args, config)
        valid = result.summary["valid"]
        test = result.summary["test"]
        source_log = Path(args.log_dir) / f"{dataset}_B0_{config['model_type']}_{config['loss_type']}_seed{args.seed}.json"
        write_json(source_log, result.summary)
        row = make_stage_row(
            dataset=dataset,
            row_kind="base",
            model_type=config["model_type"],
            block_group="B0_self",
            selected_groups=["B0_self"],
            status="completed_base",
            reason="self_only_head_selection",
            metrics=test,
            valid_metrics=valid,
            resource=group_resource(diagnostics, ["B0_self"]),
            edge_chunk_size=args.edge_chunk_size,
            dst_chunk_size=args.dst_chunk_size,
            block_dim=block_dim,
            num_blocks=len(groups["B0_self"]),
            wall_time_s=result.summary["training_time_s"],
            source_log=str(source_log),
        )
        rows.append(row)
        key = (float(valid["accuracy"]), float(valid["macro_f1"]))
        if best_base is None or key > best_base[0]:
            best_base = (key, result)
            best_config = config
    assert best_config is not None
    selected = ["B0_self"]
    current_valid = dict(best_base[1].summary["valid"])
    branch_order = [name for name in ["B1_typed", "B2_metapath", "B3_lad_scap", "B4_structure"] if name in groups]
    for group_name in branch_order:
        trial_groups = [*selected, group_name]
        trial_blocks = merge_block_groups(groups, trial_groups)
        result = _run_training(trial_blocks, graph, train_rows, valid_rows, args, best_config)
        valid = result.summary["valid"]
        test = result.summary["test"]
        improves = float(valid["accuracy"]) > float(current_valid["accuracy"]) + args.selection_epsilon_acc or float(valid["macro_f1"]) > float(current_valid["macro_f1"]) + args.selection_epsilon_f1
        decision = "kept" if improves else "dropped"
        if improves:
            selected.append(group_name)
            current_valid = dict(valid)
        gate_values = result.summary.get("block_gates", {})
        gate_value = max((float(value) for key, value in gate_values.items() if key in trial_blocks), default=0.0)
        source_log = Path(args.log_dir) / f"{dataset}_{group_name}_{best_config['model_type']}_seed{args.seed}.json"
        write_json(source_log, result.summary)
        rows.append(
            make_stage_row(
                dataset=dataset,
                row_kind="branch",
                model_type=best_config["model_type"],
                block_group=group_name,
                selected_groups=trial_groups,
                status="completed_branch",
                reason="validation_only_block_selection",
                metrics=test,
                valid_metrics=valid,
                resource=group_resource(diagnostics, trial_groups),
                edge_chunk_size=args.edge_chunk_size,
                dst_chunk_size=args.dst_chunk_size,
                block_dim=block_dim,
                num_blocks=len(trial_blocks),
                wall_time_s=result.summary["training_time_s"],
                source_log=str(source_log),
                gate_value=gate_value,
                kept_or_dropped=decision,
                drop_reason="" if decision == "kept" else "dropped_by_validation",
            )
        )
    final_blocks = merge_block_groups(groups, selected)
    final = _run_training(final_blocks, graph, train_rows, valid_rows, args, best_config)
    final_status, reason = promotion_status(dataset, final.summary["test"])
    source_log = Path(args.log_dir) / f"{dataset}_final_{best_config['model_type']}_seed{args.seed}.json"
    write_json(source_log, final.summary)
    rows.append(
        make_stage_row(
            dataset=dataset,
            row_kind="final",
            model_type=best_config["model_type"],
            block_group="final_selected",
            selected_groups=selected,
            status=final_status,
            reason=reason,
            metrics=final.summary["test"],
            valid_metrics=final.summary["valid"],
            resource=group_resource(diagnostics, selected),
            edge_chunk_size=args.edge_chunk_size,
            dst_chunk_size=args.dst_chunk_size,
            block_dim=block_dim,
            num_blocks=len(final_blocks),
            wall_time_s=float(time.perf_counter() - started),
            source_log=str(source_log),
        )
    )
    return rows


def _write_report(rows: list[dict[str, Any]], output: Path, report: Path) -> None:
    final_rows = [row for row in rows if row.get("row_kind") in {"final", "resource_guard"}]
    branch_rows = [row for row in rows if row.get("row_kind") == "branch"]
    lines = [
        "# T2-SFT-NL Safe Block Selection Seed 42",
        "",
        "No row uses logits as input, teacher logits, KD, dense P2, bounded edges, or E x d materialization.",
        "",
        "## Final Rows",
        "",
        *markdown_table(final_rows, ["dataset", "status", "accuracy", "macro_f1", "predicted_class_count", "selected_blocks", "reason"]),
        "",
        "## Block Decisions",
        "",
        *markdown_table(branch_rows, ["dataset", "block_group", "branch_valid_acc", "branch_test_acc_debug", "kept_or_dropped", "drop_reason", "gate_value"]),
        "",
        f"- CSV: `{output}`",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2 no-logits validation-only safe block selection.")
    parser.add_argument("--datasets", nargs="+", default=ALL_T2_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--medium-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--medium-hidden-dim", type=int, default=256)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--medium-block-dim", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--edge-limit", type=int, default=0)
    parser.add_argument("--medium-batch-size", type=int, default=16384)
    parser.add_argument("--scap-topk", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--selection-epsilon-acc", type=float, default=0.0)
    parser.add_argument("--selection-epsilon-f1", type=float, default=0.0)
    parser.add_argument("--run-products-full", action="store_true")
    parser.add_argument("--output", default="experiments/tables/t2_sft_safe_block_selection_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t2_sft_safe_block_selection_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/t2_sft_safe_block_selection_seed42")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        rows.extend(run_dataset_selection(dataset, args))
    output = write_csv(args.output, rows, T2_STAGE_FIELDS)
    write_json(Path(args.output).with_suffix(".json"), {"rows": rows})
    _write_report(rows, output, Path(args.report))
    print(json.dumps({"rows": len(rows), "final_rows": sum(1 for row in rows if row.get("row_kind") == "final")}, sort_keys=True))


if __name__ == "__main__":
    main()
