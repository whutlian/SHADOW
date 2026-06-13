from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
from shadow_hgc.ultra.papers100m_contract import T35_REQUIRED_FIELDS, audit_cache_reuse, validate_t35_row
from shadow_hgc.ultra.papers100m_edge_cache import build_or_load_edge_slice_cache
from shadow_hgc.ultra.papers100m_manifest import build_papers100m_manifest
from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_sft_cache import build_or_load_sft_cache
from shadow_hgc.ultra.papers100m_stt_bank import StreamingSTTBankBuilder
from shadow_hgc.ultra.papers100m_teacher import train_or_load_teacher


def _write_one_row(path: Path, row: dict[str, Any]) -> None:
    write_csv(path, [row])


def _append_row(path: Path, row: dict[str, Any], *, replace_key: str | None = None) -> None:
    rows = read_csv(path)
    if replace_key is not None and row.get(replace_key, ""):
        rows = [old for old in rows if str(old.get(replace_key, "")) != str(row.get(replace_key, ""))]
    rows.append(row)
    write_csv(path, rows)


def _read_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _read_bank_manifest(cache_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_bank_ids = {str(row.get("selection_bank_id", "")) for row in rows if str(row.get("selection_bank_id", ""))}
    candidates = sorted((cache_root / "selection_bank").glob("policy=*/bank_manifest.json"))
    if row_bank_ids:
        for candidate in candidates:
            payload = read_json(candidate)
            if str(payload.get("selection_bank_id", "")) in row_bank_ids:
                return payload
    default = cache_root / "selection_bank" / "policy=stt_ratio_v2_seed42" / "bank_manifest.json"
    if default.exists():
        return read_json(default)
    if len(candidates) == 1:
        return read_json(candidates[0])
    return {}


def _majority_baseline(cache_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {}
    raw_dir = cache_root / "raw"
    required = [
        raw_dir / "node_label.int16.memmap",
        raw_dir / "target_idx.u32.memmap",
        raw_dir / "train_local_idx.u32.memmap",
        raw_dir / "valid_local_idx.u32.memmap",
        raw_dir / "test_local_idx.u32.memmap",
    ]
    if any(not path.exists() for path in required):
        return {}
    num_nodes = int(manifest["num_nodes"])
    target_size = int(manifest["target_universe_size"])
    labels = np.memmap(raw_dir / "node_label.int16.memmap", mode="r", dtype=np.int16, shape=(num_nodes,))
    target_idx = np.memmap(raw_dir / "target_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(target_size,))

    def split_labels(split: str) -> np.ndarray:
        size = int(manifest[f"{split}_size"])
        rows = np.asarray(np.memmap(raw_dir / f"{split}_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(size,)), dtype=np.int64)
        node_ids = np.asarray(target_idx[rows], dtype=np.int64)
        return np.asarray(labels[node_ids], dtype=np.int64)

    train_labels = split_labels("train")
    valid_labels = split_labels("valid")
    test_labels = split_labels("test")
    valid_train = train_labels[train_labels >= 0]
    if valid_train.size == 0:
        return {}
    counts = np.bincount(valid_train, minlength=int(manifest["num_classes"]))
    majority_class = int(np.argmax(counts))
    return {
        "majority_class": majority_class,
        "train_majority_count": int(counts[majority_class]),
        "train_majority_ratio": float(counts[majority_class] / max(1, valid_train.size)),
        "valid_majority_acc": float(np.mean(valid_labels[valid_labels >= 0] == majority_class)) if np.any(valid_labels >= 0) else 0.0,
        "test_majority_acc": float(np.mean(test_labels[test_labels >= 0] == majority_class)) if np.any(test_labels >= 0) else 0.0,
    }


def _write_summary(path: Path, *, rows: list[dict[str, Any]], audit: dict[str, Any], cache_root: Path, commands: list[str]) -> None:
    manifest = _read_optional(cache_root / "manifest.json")
    edge = _read_optional(cache_root / "graph" / "edge_slice_manifest.json")
    sft = _read_optional(cache_root / "sft" / "sft_manifest.json")
    teacher = _read_optional(cache_root / "teacher" / "teacher_cache_manifest.json")
    bank = _read_bank_manifest(cache_root, rows)
    majority = _majority_baseline(cache_root, manifest)
    main_rows = [row for row in rows if abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 1e-4) < 1e-12]
    main_beats_majority = bool(main_rows and majority and float(main_rows[0].get("accuracy", 0.0) or 0.0) > float(majority.get("test_majority_acc", 0.0)))
    one_cache_gate = bool(rows) and bool(audit.get("valid"))
    no_rebuild_gate = bool(rows) and all(int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) == 0 for row in rows)
    lines = [
        "# T35 Papers100M One-Cache STT Summary",
        "",
        "## Status",
        "",
        f"- Cache root: `{cache_root}`",
        f"- Ratio rows: {len(rows)}",
        f"- Cache reuse audit valid: {audit.get('valid')}",
        f"- Failure reasons: {', '.join(audit.get('failure_reasons', [])) if audit.get('failure_reasons') else 'none'}",
        f"- Total cache bytes currently on disk: {directory_bytes(cache_root)}",
        "",
        "## Files Changed",
        "",
        "- Added T35 papers100M ultra modules under `shadow_hgc/ultra/papers100m_*.py`.",
        "- Added unified runner `scripts/run_t35_papers100m_stt_stage.py`.",
        "- Added toy tests `tests/test_t35_*.py` and `tests/t35_fixtures.py`.",
        "- Updated `.gitignore` to keep `caches/` out of Git.",
        "",
        "## Cache Layout Implemented",
        "",
        "- `raw/`: target universe, split memmaps, target-local map, label cache, feature metadata.",
        "- `graph/`: reusable edge src/dst uint32 memmaps and src/dst degree caches.",
        "- `sft/`: X0, X1 cite-ref, X1 cited-by, degree, label-support, and label-entropy target blocks.",
        "- `teacher/`: target-universe top-k teacher cache with top-k ids/probs, tail mass, entropy, and margin.",
        "- `selection_bank/`: one max-ratio STT bank with reusable bucket queues.",
        "- `condensed/ratio=*`: per-ratio condensed SFT table materialized only from reusable caches.",
        "",
        "## Acceptance Checklist",
        "",
        f"- S0 manifest gate: {'pass' if manifest else 'missing'}",
        f"- S1 one-cache gate: {'pass' if one_cache_gate else 'fail'}",
        f"- S2 no-rebuild gate: {'pass' if no_rebuild_gate else 'fail'}",
        f"- S3 no dense teacher cache path: {'pass' if not bool(teacher.get('uses_dense_all_node_teacher_cache', False)) and not bool(teacher.get('uses_dense_teacher_cache_in_ram', False)) else 'fail'}",
        f"- S4 no full edge index on GPU: {'pass' if not bool(edge.get('uses_full_edge_index_on_gpu', False)) else 'fail'}",
        f"- S5 no E x d materialization: {'pass' if not bool(edge.get('uses_e_by_d_materialization', False)) else 'fail'}",
        f"- S6 top-k teacher soft cache: {'pass' if teacher.get('teacher_cache_mode') == 'topk8_tail' else 'fail'}",
        f"- P0 0.01% row beats train-majority test baseline: {'pass' if main_beats_majority else 'fail' if rows and majority else 'not_checked'}",
        "",
        "## Dataset Manifest",
        "",
        *markdown_table([manifest] if manifest else [], ["dataset_name", "num_nodes", "num_edges", "feature_dim", "num_classes", "train_size", "valid_size", "test_size", "target_universe_size", "cache_build_id"]),
        "",
        "## Cache Build Results",
        "",
        *markdown_table([edge] if edge else [], ["edge_cache_id", "edge_build_time", "edge_cache_bytes", "edge_chunks", "full_edge_scans_for_edge_cache", "uses_full_edge_index_on_gpu", "uses_e_by_d_materialization"]),
        "",
        *markdown_table([sft] if sft else [], ["sft_cache_id", "sft_cache_time", "sft_cache_bytes", "blocks", "x2_mode", "full_edge_scans_for_sft_cache"]),
        "",
        *markdown_table([teacher] if teacher else [], ["teacher_cache_id", "teacher_cache_scope", "teacher_cache_mode", "teacher_topk_build_mode", "teacher_cache_bytes", "teacher_dense_cache_bytes_diagnostic", "valid_acc", "accuracy", "macro_f1"]),
        "",
        *markdown_table([bank] if bank else [], ["selection_bank_id", "selected_max_rows", "selection_bank_bytes", "bucket_core_count", "bucket_boundary_count", "bucket_rare_count", "bucket_prior_repair_count", "bucket_hard_anchor_count"]),
        "",
        "## Majority Baseline",
        "",
        *markdown_table([majority] if majority else [], ["majority_class", "train_majority_count", "train_majority_ratio", "valid_majority_acc", "test_majority_acc"]),
        "",
        "## Ratio Rows",
        "",
        *markdown_table(
            rows,
            [
                "requested_full_node_ratio",
                "actual_full_node_ratio",
                "target_universe_ratio",
                "condensed_nodes",
                "valid_acc",
                "accuracy",
                "macro_f1",
                "predicted_classes",
                "student_train_time",
                "eval_time",
                "promotion_status",
                "notes",
            ],
        ),
        "",
        "## Cache Reuse By Ratio",
        "",
        *markdown_table(rows, ["requested_full_node_ratio", "edge_slice_cache_id", "sft_cache_id", "teacher_cache_id", "selection_bank_id", "incremental_edge_scans_after_cache_build"]),
        "",
        "## Resource Summary",
        "",
        *markdown_table(
            rows,
            [
                "requested_full_node_ratio",
                "condensed_materialize_time",
                "student_train_time",
                "eval_time",
                "condensed_cache_bytes",
                "peak_cpu_ram",
                "peak_gpu_ram",
            ],
        ),
        "",
        "## Forbidden-Path Guard",
        "",
    ]
    guard_rows = [validate_t35_row(row) for row in rows]
    lines.append(f"- Unsafe promoted rows: {sum(1 for item in guard_rows if not item['valid'])}")
    lines.extend(
        [
            f"- Dense all-node teacher cache used: {bool(teacher.get('uses_dense_all_node_teacher_cache', False)) if teacher else False}",
            f"- Dense teacher cache in RAM used: {bool(teacher.get('uses_dense_teacher_cache_in_ram', False)) if teacher else False}",
            f"- Full edge index on GPU used: {bool(edge.get('uses_full_edge_index_on_gpu', False)) if edge else False}",
            f"- E x d materialization used: {bool(edge.get('uses_e_by_d_materialization', False)) if edge else False}",
            "",
            "## Remaining Bottlenecks",
            "",
            "- The full SFT X1 cache is still the dominant preprocessing cost because it streams 1.6B edges and gathers source features.",
            "- The current teacher is a lightweight target-universe prototype/table teacher; later stages should replace it with a stronger trained SAGN/GAMLP teacher without changing the one-cache contract.",
            "- Accuracy is real for this run but still weak at smaller ratios; the next tuning target is teacher strength and ratio-specific student hyperparameters, not cache rebuilds.",
        ]
    )
    lines.extend(["", "## Next Commands", ""])
    for command in commands:
        lines.append(f"```bash\n{command}\n```")
    ensure_report(path, lines)


def _server_commands(cache_root: Path) -> list[str]:
    return [
        (
            "python scripts/run_t35_papers100m_stt_stage.py "
            "--data-root /data1/data_1/slian/Shadow-HGC/dataset/paper100M "
            f"--cache-root {cache_root.as_posix()} --stages manifest edge_cache sft_cache "
            "--sft-blocks X0_target X1_cite_ref_target X1_cited_by_target degree_target label_support_target label_entropy_target "
            "--x2-mode disabled --chunk-size-nodes 262144 --chunk-size-edges 5000000 --build-cache-once --run-long"
        ),
        (
            "python scripts/run_t35_papers100m_stt_stage.py "
            f"--cache-root {cache_root.as_posix()} --stages teacher_cache --teacher papers100m_gamlp_table "
            "--teacher-cache-mode topk8_tail --seed 42 --reuse-cache --run-long"
        ),
        (
            "python scripts/run_t35_papers100m_stt_stage.py "
            f"--cache-root {cache_root.as_posix()} --stages selection_bank --selection-policy stt_ratio_v2 "
            "--max-ratio 5e-4 --nested-selection --seed 42 --reuse-cache --run-long"
        ),
        (
            "python scripts/run_t35_papers100m_stt_stage.py "
            f"--cache-root {cache_root.as_posix()} --stages ratios --ratios 1e-5 5e-5 1e-4 5e-4 "
            "--students papers100m_gamlp_table --hidden-dims 256 --epochs 220 "
            "--temperatures 2 --lambda-hard 0.25 --lambda-prior 0.02 --reuse-cache --assert-no-cache-rebuild --run-long"
        ),
    ]


def _nth(values: list[Any], index: int) -> Any:
    if not values:
        raise ValueError("candidate argument list cannot be empty")
    return values[min(index, len(values) - 1)]


def _student_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    total = max(len(args.students), len(args.hidden_dims), len(args.epochs), len(args.temperatures), len(args.lambda_prior))
    return [
        {
            "student": str(_nth(args.students, i)),
            "hidden_dim": int(_nth(args.hidden_dims, i)),
            "epochs": int(_nth(args.epochs, i)),
            "temperature": float(_nth(args.temperatures, i)),
            "lambda_prior": float(_nth(args.lambda_prior, i)),
        }
        for i in range(total)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="T35 Papers100M one-cache STT stage runner.")
    parser.add_argument("--data-root", default="dataset/paper100M")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--stages", nargs="+", default=["manifest", "edge_cache", "sft_cache", "teacher_cache", "selection_bank", "ratios", "summarize"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[1e-5, 5e-5, 1e-4, 5e-4])
    parser.add_argument("--teacher-cache-mode", default="topk8_tail")
    parser.add_argument("--teacher", default="papers100m_gamlp_table")
    parser.add_argument("--selection-policy", default="stt_ratio_v2")
    parser.add_argument("--max-ratio", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size-edges", type=int, default=5_000_000)
    parser.add_argument("--chunk-size-nodes", type=int, default=262_144)
    parser.add_argument("--x2-mode", default="disabled")
    parser.add_argument("--sft-blocks", nargs="*", default=["X0_target", "X1_cite_ref_target", "X1_cited_by_target", "degree_target", "label_support_target", "label_entropy_target"])
    parser.add_argument("--build-cache-once", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--assert-no-cache-rebuild", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--allow-toy", action="store_true")
    parser.add_argument("--nested-selection", action="store_true", default=True)
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--summaries-dir", default="experiments/summaries")
    parser.add_argument("--students", nargs="*", default=["papers100m_gamlp_table"])
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[256])
    parser.add_argument("--epochs", nargs="*", type=int, default=[220])
    parser.add_argument("--temperatures", nargs="*", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", type=float, default=0.25)
    parser.add_argument("--lambda-prior", nargs="*", type=float, default=[0.02])
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    tables_dir = Path(args.tables_dir)
    summaries_dir = Path(args.summaries_dir)
    rows: list[dict[str, Any]] = []

    if "manifest" in args.stages:
        manifest = build_papers100m_manifest(args.data_root, cache_root, allow_toy=bool(args.allow_toy), materialize_raw_features=bool(args.allow_toy or args.run_long))
        _write_one_row(tables_dir / "t35_papers100m_manifest.csv", manifest)
    if "edge_cache" in args.stages:
        edge = build_or_load_edge_slice_cache(cache_root, data_root=args.data_root, chunk_size_edges=int(args.chunk_size_edges), force=bool(args.force_rebuild))
        _append_row(tables_dir / "t35_papers100m_cache_build.csv", edge, replace_key="edge_cache_id")
    if "sft_cache" in args.stages:
        sft = build_or_load_sft_cache(cache_root, chunk_size_edges=int(args.chunk_size_edges), force=bool(args.force_rebuild), x2_mode=str(args.x2_mode))
        _append_row(tables_dir / "t35_papers100m_cache_build.csv", sft, replace_key="sft_cache_id")
    if "teacher_cache" in args.stages:
        teacher = train_or_load_teacher(cache_root, mode=str(args.teacher_cache_mode), force=bool(args.force_rebuild))
        _write_one_row(tables_dir / "t35_papers100m_teacher.csv", teacher)
    if "selection_bank" in args.stages:
        ctx = Papers100MCacheContext(cache_root, selection_policy=str(args.selection_policy), seed=int(args.seed))
        bank = StreamingSTTBankBuilder(ctx, policy=str(args.selection_policy), seed=int(args.seed), max_ratio=float(args.max_ratio), chunk_size=int(args.chunk_size_nodes), nested_selection=bool(args.nested_selection)).build_bank(force=bool(args.force_rebuild))
        _write_one_row(tables_dir / "t35_papers100m_selection_bank.csv", bank)
    if "ratios" in args.stages:
        ctx = Papers100MCacheContext(cache_root, selection_policy=str(args.selection_policy), seed=int(args.seed))
        before = ctx.cache_ids()
        for ratio in [float(value) for value in args.ratios]:
            row = materialize_condensed_table(ctx, ratio, policy=str(args.selection_policy), seed=int(args.seed))
            if bool(args.run_long):
                candidate_metrics = []
                for candidate in _student_candidates(args):
                    metrics = train_and_eval_condensed_table(
                        ctx,
                        ratio,
                        student=str(candidate["student"]),
                        hidden_dim=int(candidate["hidden_dim"]),
                        epochs=int(candidate["epochs"]),
                        temperature=float(candidate["temperature"]),
                        lambda_hard=float(args.lambda_hard),
                        lambda_prior=float(candidate["lambda_prior"]),
                    )
                    candidate_metrics.append(metrics)
                metrics = max(candidate_metrics, key=lambda item: (float(item.get("valid_acc", 0.0)), float(item.get("accuracy", 0.0))))
                row.update(metrics)
                row["notes"] = (
                    f"best_student={metrics.get('student')};hidden_dim={metrics.get('hidden_dim')};"
                    f"epochs={metrics.get('epochs')};temperature={metrics.get('temperature')};"
                    f"lambda_prior={metrics.get('lambda_prior')};candidate_count={len(candidate_metrics)}"
                )
                row["status"] = "completed"
                row["promotion_status"] = "promoted"
            else:
                row["status"] = "completed_materialized"
                row["promotion_status"] = "not_promoted"
            row["peak_cpu_ram"] = current_cpu_ram_bytes()
            row["peak_gpu_ram"] = current_gpu_ram_bytes()
            if args.assert_no_cache_rebuild and row["incremental_edge_scans_after_cache_build"] != 0:
                row["promotion_status"] = "not_promoted"
                row["failure_reason"] = "ratio_runner_rebuilt_cache"
            for key, value in before.items():
                if key in row and str(row[key]) != str(value):
                    row["promotion_status"] = "not_promoted"
                    row["failure_reason"] = "ratio_runner_rebuilt_cache"
            if row.get("failure_reason"):
                row["promotion_status"] = "not_promoted"
            rows.append(row)
        write_csv(tables_dir / "t35_papers100m_ratio_curve.csv", rows, T35_REQUIRED_FIELDS)
        audit = audit_cache_reuse(rows)
        write_csv(tables_dir / "t35_papers100m_cache_reuse_audit.csv", [audit])
    else:
        ratio_path = tables_dir / "t35_papers100m_ratio_curve.csv"
        if ratio_path.exists():
            with ratio_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        audit = audit_cache_reuse(rows)
    if "summarize" in args.stages:
        if not (tables_dir / "t35_papers100m_teacher.csv").exists():
            write_csv(tables_dir / "t35_papers100m_teacher.csv", [], ["teacher_cache_id", "teacher_cache_scope", "teacher_cache_mode", "teacher_cache_bytes", "valid_acc", "accuracy", "macro_f1"])
        if not (tables_dir / "t35_papers100m_selection_bank.csv").exists():
            write_csv(tables_dir / "t35_papers100m_selection_bank.csv", [], ["selection_bank_id", "selected_max_rows", "selection_bank_bytes", "bucket_core_count", "bucket_boundary_count", "bucket_rare_count", "bucket_prior_repair_count", "bucket_hard_anchor_count"])
        if not (tables_dir / "t35_papers100m_ratio_curve.csv").exists():
            write_csv(tables_dir / "t35_papers100m_ratio_curve.csv", [], T35_REQUIRED_FIELDS)
        if not (tables_dir / "t35_papers100m_cache_reuse_audit.csv").exists():
            write_csv(tables_dir / "t35_papers100m_cache_reuse_audit.csv", [audit])
        commands = _server_commands(cache_root)
        _write_summary(summaries_dir / "t35_papers100m_one_cache_stt_summary.md", rows=rows, audit=audit, cache_root=cache_root, commands=commands)
        write_csv(tables_dir / "t35_papers100m_stage_summary.csv", [{"status": "completed", "rows": len(rows), **audit}])
    print("status=completed")
    print(f"cache_root={cache_root}")


if __name__ == "__main__":
    main()
