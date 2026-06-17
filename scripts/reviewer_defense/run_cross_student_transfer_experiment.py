from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.run_t39_unified_e2e_stage import (  # noqa: E402
    ALIASES,
    DEFAULT_RATIOS,
    MEDIUM_BLOCKS,
    _condensed_nodes,
    _directory_bytes,
    _exception_reason,
    _labels_for_medium,
    _manifest_for_medium,
    _manifest_shared_cache_time,
    _teacher_for_medium,
)
from scripts.run_t41_domain_transport_finalization import (  # noqa: E402
    _candidate_schedule,
    _domain_stats_for_medium,
    _selected_transport_metadata,
)
from scripts.t24_common import ensure_report, markdown_table, write_csv  # noqa: E402
from shadow_hgc.sft.unified_objective import select_unified_prefixes_from_memmap  # noqa: E402
from shadow_hgc.sft.unified_stt import NUM_CLASSES, fvalue  # noqa: E402
from shadow_hgc.train.lazy_sft_memmap import train_lazy_sft_from_memmap  # noqa: E402


STUDENT_CHOICES = ("stt_gated_mixer", "mlp", "linear_probe", "sagn_like", "gamlp_like")
DEFAULT_DATASETS = ("Reddit", "ogbn-products")

FIELDS = [
    "dataset",
    "ratio",
    "budget",
    "artifact_method",
    "selection_policy",
    "reservoir_mode",
    "student",
    "model_type",
    "student_internal_style",
    "hidden_dim",
    "epochs",
    "seed",
    "status",
    "valid_acc",
    "test_acc",
    "macro_f1",
    "relative_to_stt_on_same_artifact",
    "predicted_classes",
    "trainable_params",
    "selection_time_sec",
    "train_time_sec",
    "eval_time_sec",
    "post_cache_time_sec",
    "shared_cache_time_sec",
    "storage_bytes",
    "peak_cpu_ram_gb",
    "peak_gpu_mem_gb",
    "uses_teacher_soft_targets",
    "uses_teacher_probs_as_input_features",
    "failure_reason",
    "selected_blocks_json",
]


def _canonical_datasets(values: list[str]) -> list[str]:
    out = [ALIASES[str(value)] for value in values]
    if out == ["all"] or "all" in out:
        return ["ogbn-arxiv", "Reddit", "ogbn-products"]
    unsupported = [value for value in out if value == "ogbn-papers100M"]
    if unsupported:
        raise SystemExit("cross-student transfer currently targets medium SFT memmap datasets; papers100M table students are handled separately")
    return out


def _ratios_for_dataset(args: argparse.Namespace, dataset: str) -> list[float]:
    if args.ratios:
        return [float(value) for value in args.ratios]
    return [float(value) for value in DEFAULT_RATIOS[dataset]]


def _student_config(student: str, schedule: Any, args: argparse.Namespace) -> dict[str, Any]:
    student = str(student)
    hidden_dim = int(args.hidden_dim_override or schedule.hidden_dim)
    common = {
        "hidden_dim": hidden_dim,
        "dropout": float(args.dropout),
        "num_layers": int(args.num_layers),
        "label_dropout": float(args.label_dropout),
        "block_dropout": 0.0,
        "hop_dropout": 0.0,
        "student_internal_style": "cross_student",
    }
    if student == "stt_gated_mixer":
        return {
            **common,
            "model_type": "stt_gated_mixer",
            "student_internal_style": schedule.student_internal_style,
        }
    if student == "mlp":
        return {**common, "model_type": "concat_mlp", "student_internal_style": "concat_mlp"}
    if student == "linear_probe":
        return {
            **common,
            "model_type": "linear_probe",
            "student_internal_style": "concat_linear",
            "num_layers": 1,
            "dropout": 0.0,
            "label_dropout": 0.0,
        }
    if student == "sagn_like":
        return {**common, "model_type": "sagn_lite_v2", "student_internal_style": "sagn_like"}
    if student == "gamlp_like":
        return {**common, "model_type": "gamlp_lite_v2", "student_internal_style": "gamlp_like"}
    raise ValueError(f"unknown student: {student}")


def _blocked_row(
    *,
    dataset: str,
    ratio: float,
    budget: int,
    student: str,
    policy: str,
    reservoir_mode: str,
    seed: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "ratio": float(ratio),
        "budget": int(budget),
        "artifact_method": "Shadow-HGC-STT-U",
        "selection_policy": policy,
        "reservoir_mode": reservoir_mode,
        "student": student,
        "seed": int(seed),
        "status": "blocked",
        "failure_reason": str(reason),
    }


def run_dataset(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    ratios = _ratios_for_dataset(args, dataset)
    print(json.dumps({"event": "dataset_start", "dataset": dataset, "ratios": ratios}, sort_keys=True), flush=True)
    labels, train_rows, valid_rows, test_rows = _labels_for_medium(dataset, args)
    manifest_dir = _manifest_for_medium(dataset, args)
    teacher_probs_path, teacher_valid_acc, teacher_bytes = _teacher_for_medium(dataset, args)
    if dataset != "Reddit":
        teacher_probs_path = None
        teacher_valid_acc = None
        teacher_bytes = 0
    domain_buckets, domain_gap = _domain_stats_for_medium(dataset, manifest_dir, labels, train_rows, int(args.seed))
    budgets = [_condensed_nodes(dataset, ratio) for ratio in ratios]
    max_budget = max(budgets)
    policy = str(args.selection_policy)
    max_schedule = _candidate_schedule(dataset, max_budget, teacher_valid_acc, domain_gap, args, policy)
    stage_selection_weights = {
        int(budget): dict(_candidate_schedule(dataset, int(budget), teacher_valid_acc, domain_gap, args, policy).selection_weights)
        for budget in budgets
    }
    selection_started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "selection_start",
                "dataset": dataset,
                "policy": policy,
                "max_budget": max_budget,
                "reservoir_mode": args.reservoir_mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    prefixes = select_unified_prefixes_from_memmap(
        labels=labels,
        train_rows=train_rows,
        manifest_dir=manifest_dir,
        budgets=budgets,
        num_classes=NUM_CLASSES[dataset],
        seed=int(args.seed),
        selection_weights=max_schedule.selection_weights,
        stage_selection_weights=stage_selection_weights if str(args.reservoir_mode) == "staged" else None,
        teacher_probs_path=teacher_probs_path,
        domain_bucket_ids=domain_buckets,
    )
    selection_time = float(time.perf_counter() - selection_started)
    print(json.dumps({"event": "selection_done", "dataset": dataset, "selection_time_sec": selection_time}, sort_keys=True), flush=True)
    teacher_probs = np.load(teacher_probs_path, mmap_mode="r") if teacher_probs_path and bool(args.use_soft_targets) else None
    cache_bytes = _directory_bytes(manifest_dir) + int(teacher_bytes)
    shared_cache_time_sec = _manifest_shared_cache_time(manifest_dir)
    rows: list[dict[str, Any]] = []
    selected_by_budget: dict[int, torch.Tensor] = {}
    for ratio, budget in zip(ratios, budgets):
        base_selected_rows = prefixes[int(budget)]
        selected_rows, _diag = _selected_transport_metadata(
            policy=policy,
            selected_rows=base_selected_rows,
            base_selected_rows=base_selected_rows,
            labels=labels,
            train_rows=train_rows,
            domain_buckets=domain_buckets,
            num_classes=NUM_CLASSES[dataset],
            budget=int(budget),
            domain_gap_train_all=domain_gap,
            enabled=bool(args.enable_domain_transport),
            seed=int(args.seed),
        )
        selected_by_budget[int(budget)] = selected_rows
        for student in args.students:
            student = str(student)
            schedule = _candidate_schedule(dataset, int(budget), teacher_valid_acc, domain_gap, args, policy)
            cfg = _student_config(student, schedule, args)
            epochs = int(args.epochs_override or schedule.epochs)
            if int(args.epochs_cap) > 0:
                epochs = min(epochs, int(args.epochs_cap))
            print(
                json.dumps(
                    {"event": "student_start", "dataset": dataset, "ratio": ratio, "budget": budget, "student": student, "epochs": epochs},
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                result = train_lazy_sft_from_memmap(
                    manifest_dir=manifest_dir,
                    labels=labels,
                    train_rows=selected_by_budget[int(budget)],
                    valid_rows=valid_rows,
                    test_rows=test_rows,
                    num_classes=NUM_CLASSES[dataset],
                    device=args.device,
                    model_type=str(cfg["model_type"]),
                    hidden_dim=int(cfg["hidden_dim"]),
                    dropout=float(cfg["dropout"]),
                    student_internal_style=str(cfg["student_internal_style"]),
                    num_layers=int(cfg["num_layers"]),
                    block_dropout=float(cfg["block_dropout"]),
                    hop_dropout=float(cfg["hop_dropout"]),
                    label_dropout=float(cfg["label_dropout"]),
                    selected_blocks=MEDIUM_BLOCKS[dataset],
                    loss_type=str(args.loss_type),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(args.seed),
                    eval_every=int(args.eval_every),
                    teacher_probs=teacher_probs,
                    lambda_hard=float(schedule.loss_weights["lambda_hard"]),
                    lambda_soft=float(schedule.loss_weights["lambda_soft"]) if teacher_probs is not None else 0.0,
                    lambda_prior=float(schedule.loss_weights["lambda_prior"]) if teacher_probs is not None else 0.0,
                    soft_temperature=float(schedule.soft_temperature),
                ).summary
                test = result["test"]
                valid = result["valid"]
                row = {
                    "dataset": dataset,
                    "ratio": float(ratio),
                    "budget": int(budget),
                    "artifact_method": "Shadow-HGC-STT-U",
                    "selection_policy": policy,
                    "reservoir_mode": str(args.reservoir_mode),
                    "student": student,
                    "model_type": str(result.get("model_type", cfg["model_type"])),
                    "student_internal_style": str(result.get("student_internal_style", cfg["student_internal_style"])),
                    "hidden_dim": int(cfg["hidden_dim"]),
                    "epochs": int(result.get("epochs_ran", epochs)),
                    "seed": int(args.seed),
                    "status": "completed",
                    "valid_acc": float(valid.get("accuracy", 0.0)),
                    "test_acc": float(test.get("accuracy", 0.0)),
                    "macro_f1": float(test.get("macro_f1", 0.0)),
                    "predicted_classes": int(test.get("predicted_class_count", 0)),
                    "trainable_params": int(result.get("trainable_params", 0)),
                    "selection_time_sec": selection_time,
                    "train_time_sec": float(result.get("training_time_s", 0.0)),
                    "eval_time_sec": float(result.get("inference_time_s", 0.0)),
                    "post_cache_time_sec": float(selection_time + fvalue(result.get("training_time_s")) + fvalue(result.get("inference_time_s"))),
                    "shared_cache_time_sec": shared_cache_time_sec,
                    "storage_bytes": int(cache_bytes),
                    "peak_cpu_ram_gb": float(result.get("peak_cpu_ram_gb", 0.0)),
                    "peak_gpu_mem_gb": float(result.get("peak_gpu_ram_gb", 0.0)),
                    "uses_teacher_soft_targets": bool(teacher_probs is not None),
                    "uses_teacher_probs_as_input_features": False,
                    "failure_reason": "",
                    "selected_blocks_json": json.dumps(MEDIUM_BLOCKS[dataset]),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "student_done",
                            "dataset": dataset,
                            "ratio": ratio,
                            "student": student,
                            "accuracy": row["test_acc"],
                            "macro_f1": row["macro_f1"],
                            "train_time_sec": row["train_time_sec"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as exc:
                reason = _exception_reason(exc)
                rows.append(
                    _blocked_row(
                        dataset=dataset,
                        ratio=float(ratio),
                        budget=int(budget),
                        student=student,
                        policy=policy,
                        reservoir_mode=str(args.reservoir_mode),
                        seed=int(args.seed),
                        reason=reason,
                    )
                )
                print(json.dumps({"event": "student_blocked", "dataset": dataset, "ratio": ratio, "student": student, "reason": reason}, sort_keys=True), flush=True)
                if bool(args.fail_fast):
                    raise
    return rows


def _add_relative_to_stt(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, float, int], float] = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("student") == "stt_gated_mixer":
            baselines[(str(row["dataset"]), float(row["ratio"]), int(row["seed"]))] = float(row["test_acc"])
    for row in rows:
        key = (str(row.get("dataset")), float(row.get("ratio", 0.0) or 0.0), int(row.get("seed", 0) or 0))
        base = baselines.get(key)
        if base is not None and row.get("status") == "completed":
            row["relative_to_stt_on_same_artifact"] = float(row["test_acc"]) - float(base)
        else:
            row["relative_to_stt_on_same_artifact"] = ""


def write_summary(rows: list[dict[str, Any]], path: str | Path) -> Path:
    completed = [row for row in rows if row.get("status") == "completed"]
    compact = []
    for row in completed:
        compact.append(
            {
                "dataset": row["dataset"],
                "ratio": row["ratio"],
                "student": row["student"],
                "test_acc": f"{float(row['test_acc']):.6f}",
                "macro_f1": f"{float(row['macro_f1']):.6f}",
                "delta_vs_stt": f"{float(row['relative_to_stt_on_same_artifact']):+.6f}" if row["relative_to_stt_on_same_artifact"] != "" else "",
                "params": row["trainable_params"],
                "train_s": f"{float(row['train_time_sec']):.2f}",
                "eval_s": f"{float(row['eval_time_sec']):.2f}",
            }
        )
    blocked = [row for row in rows if row.get("status") != "completed"]
    lines = [
        "# Cross-student transfer experiment",
        "",
        "This table trains different student heads on the same Shadow-HGC-STT-U condensed SFT table rows. The experiment tests whether the artifact is a graph-signal table tailored to a student family, rather than an architecture-agnostic synthetic graph.",
        "",
        "## Accuracy and transfer gap",
        "",
        *markdown_table(compact, ["dataset", "ratio", "student", "test_acc", "macro_f1", "delta_vs_stt", "params", "train_s", "eval_s"]),
        "",
        "## Blocked rows",
        "",
        *markdown_table(blocked, ["dataset", "ratio", "student", "status", "failure_reason"]),
    ]
    return ensure_report(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-student transfer on one Shadow-HGC-STT-U SFT artifact.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--students", nargs="+", choices=list(STUDENT_CHOICES), default=list(STUDENT_CHOICES))
    parser.add_argument("--selection-policy", default="domain_transport")
    parser.add_argument("--reservoir-mode", choices=["staged", "legacy"], default="staged")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-cache-budget-mb", type=int, default=256)
    parser.add_argument("--arxiv-manifest-dir", default="experiments/preprop/t22_ogbn_arxiv_seed42")
    parser.add_argument("--products-manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--reddit-manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--arxiv-dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--products-dataset-root", default="dataset/ogbn_products")
    parser.add_argument("--reddit-memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--use-reddit-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reddit-teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--use-soft-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-domain-transport", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-dim-override", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0)
    parser.add_argument("--epochs-cap", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=1000000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-csv", default="experiments/tables/cross_student_transfer_seed42.csv")
    parser.add_argument("--summary", default="experiments/summaries/cross_student_transfer_summary.md")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset in _canonical_datasets(args.datasets):
        rows.extend(run_dataset(args, dataset))
    _add_relative_to_stt(rows)
    csv_path = write_csv(args.output_csv, rows, fieldnames=FIELDS)
    summary_path = write_summary(rows, args.summary)
    print(json.dumps({"status": "completed", "csv": str(csv_path), "summary": str(summary_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
